import torch
import argparse
from model import smolTransformer
from data import DataLoaderLite
import time
import math
from utils import init_distributed_mode


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--global_batch_size", type=int, default=2**19) # 2**19 ~500k tokens per batch
    parser.add_argument("--micro_batch_size", type=int, default=8) # 16
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50)

    return parser


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."


    # --- reproducibility ---
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)


    # --- distributed training setup ---
    # init_distributed_mode(args)
    args.device = "cuda"


    # --- load data ---
    train_loader = DataLoaderLite(B=args.micro_batch_size, T=args.sequence_length)
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * args.sequence_length)


    # --- initialize model ---
    model = smolTransformer().to(args.device)
    # model = torch.compile(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f}M")


    # --- optimizer, criterion, gradient scaler for fp16 ---
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2] # weights and embeddings 
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2] # biases and layernorms

    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': args.weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True
    )

    criterion_train = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")


    # --- learning rate scheduler with warmup and cosine decay ---
    def lr_lambda(step):
        max_lr = args.lr
        min_lr = max_lr * 0.1

        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps

        elif step > args.max_steps:
            return min_lr / max_lr

        else:
            decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (min_lr + coeff * (max_lr - min_lr)) / max_lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # --- training loop ---
    for i in range(args.max_steps):
        t0 = time.time()
        model.train()

        loss_accum = 0.0

        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(args.device), y.to(args.device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion_train(logits.view(-1, logits.size(-1)), y.view(-1))

            loss = loss / grad_accum_steps
            loss_accum += loss.item()

            scaler.scale(loss).backward() # loss * S → gradients * S

        scaler.unscale_(optimizer) # unscale the gradients, i.e. divide by S
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer) # unscale if not already done + check for inf/NaN in gradients
        scaler.update() # dynamically adjust the scaling factor S
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.time()
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
        print(f"Step {i}: loss = {loss_accum:.4f}, lr = {optimizer.param_groups[0]['lr']:.6f}, grad_norm = {norm:.4f}, time = {(t1 - t0)*1000:.4f}ms, tokens/sec = {tokens_per_sec:.2f}")

        lr_scheduler.step()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
