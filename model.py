import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    sequence_length: int
    num_layers: int
    num_heads: int
    embed_dim: int
    vocab_size: int = 50304 # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token = 50257 tokens but we use 50304 because it slightly speeds up training (contains more powers of two)


class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim, max_sequence_length, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_sequence_length, dtype=inv_freq.dtype)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", angles.cos()[None, None, :, :], persistent=False) # persistent=False means this buffer won't be saved in the state dict
        self.register_buffer("sin_cached", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, sequence_length):
        assert 0 <= sequence_length <= self.cos_cached.size(2), "Sequence length exceeds RoPE cache bounds"
        return self.cos_cached[:, :, :sequence_length, :], self.sin_cached[:, :, :sequence_length, :]


def apply_rotary_embedding(x, cos_sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos, sin = cos_sin
    cos, sin = cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim

        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.output_proj.NANOGPT_SCALE_INIT = 1 # flag this layer for special initialization scaling

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, T, C) -> (B, num_heads, T, C // num_heads)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        q = apply_rotary_embedding(q, cos_sin)
        k = apply_rotary_embedding(k, cos_sin)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, num_heads, T, C // num_heads)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        attn = self.output_proj(attn)

        return attn


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.down_proj = nn.Linear(4 * config.embed_dim, config.embed_dim)

        self.down_proj.NANOGPT_SCALE_INIT = 1 # flag this layer for special initialization scaling

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x).square()
        x = self.down_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rms_1 = nn.RMSNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = nn.RMSNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(self.rms_1(x), cos_sin)
        x = x + self.mlp(self.rms_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.embed_dim)
        self.rotary_emb = RotaryEmbedding(config.embed_dim // config.num_heads, config.sequence_length)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.rms_f = nn.RMSNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.lm_head.weight = self.wte.weight # weight sharing scheme

        self.apply(self._init_weights) # init params, apply iterates over all submodules of this module

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.num_layers) ** -0.5   # 2 * self.config.num_layers is the number of layers in the residual stream
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.sequence_length, f"Cannot forward sequence of length {T}, block size is only {self.config.sequence_length}"

        x = self.wte(idx)
        cos_sin = self.rotary_emb(T)

        for block in self.blocks:
            x = block(x, cos_sin)

        x = self.rms_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits


def smolTransformer(sequence_length):
    config = Config(
        sequence_length=sequence_length,
        num_layers=12,
        num_heads=12,
        embed_dim=768,
    )
    return Transformer(config)
