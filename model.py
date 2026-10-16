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


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        assert config.embed_dim % config.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim

        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.output_proj.NANOGPT_SCALE_INIT = 1 # flag this layer for special initialization scaling

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, T, C) -> (B, num_heads, T, C // num_heads)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, num_heads, T, C // num_heads)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        attn = self.output_proj(attn)

        return attn


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(4 * config.embed_dim, config.embed_dim)

        self.down_proj.NANOGPT_SCALE_INIT = 1 # flag this layer for special initialization scaling

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.embed_dim)
        self.wpe = nn.Embedding(config.sequence_length, config.embed_dim)

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])

        self.ln_f = nn.LayerNorm(config.embed_dim)
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

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.wpe(pos) # position embeddings of shape (T, embed_dim)
        tok_emb = self.wte(idx) # token embeddings of shape (B, T, embed_dim)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
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
