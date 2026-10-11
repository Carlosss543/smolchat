import torch
import argparse
from contextlib import nullcontext
from model import smolTransformer
from data import make_loader
import time
import math
from utils import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--global_batch_size", type=int, default=2**19) # 2**19 ~500k tokens per batch
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup_steps", type=int, default=715) # TRY TO REDUCE IT LATER
    parser.add_argument("--max_steps", type=int, default=None)

    return parser


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."


    # --- reproducibility ---
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)


    # --- distributed training setup ---
    init_distributed_mode(args)


    # --- load data ---
    train_loader, _ = make_loader(path_to_dir=args.data_dir, split="train", batch_size=args.micro_batch_size, sequence_length=args.sequence_length, distributed=args.distributed, rank=args.ddp_rank, world_size=args.ddp_world_size)

    assert args.global_batch_size % (args.micro_batch_size * args.sequence_length * args.ddp_world_size) == 0, "Global batch size must be divisible by micro batch size * sequence length * number of distributed processes"
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * args.sequence_length * args.ddp_world_size)


    # --- initialize model ---
    model = smolTransformer().to(args.device)

    model = torch.compile(model)

    if args.distributed:
        model = DDP(model, device_ids=[args.ddp_local_rank])

    if args.master_process:
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
    max_steps = args.max_steps if args.max_steps is not None else len(train_loader) // grad_accum_steps
    if args.master_process:
        print(f"max_steps = {max_steps}, grad_accum_steps = {grad_accum_steps}")

    def lr_lambda(step):
        max_lr = args.lr
        min_lr = max_lr * 0.1

        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps

        elif step > max_steps:
            return min_lr / max_lr

        else:
            decay_ratio = (step - args.warmup_steps) / (max_steps - args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (min_lr + coeff * (max_lr - min_lr)) / max_lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # --- training loop ---
    model.train()
    train_iter = iter(train_loader)

    for step in range(max_steps):
        t0 = time.time()
        
        loss_accum = torch.tensor(0.0, device=args.device)

        for micro_step in range(grad_accum_steps):
            x, y = next(train_iter)
            x, y = x.to(args.device), y.to(args.device)

            sync_context = (model.no_sync() if args.distributed and (micro_step+1) < grad_accum_steps else nullcontext())

            with sync_context:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion_train(logits.view(-1, logits.size(-1)), y.view(-1))

                loss = loss / grad_accum_steps
                loss_accum += loss.detach()

                scaler.scale(loss).backward() # loss * S → gradients * S

        scaler.unscale_(optimizer) # unscale the gradients, i.e. divide by S
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer) # unscale if not already done + check for inf/NaN in gradients
        scaler.update() # dynamically adjust the scaling factor S
        optimizer.zero_grad(set_to_none=True)

        if args.distributed:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_sec = (args.micro_batch_size * args.sequence_length * grad_accum_steps * args.ddp_world_size) / dt

        if args.master_process:
            print(f"Step {step}: loss = {loss_accum:.4f}, lr = {optimizer.param_groups[0]['lr']:.4e}, grad_norm = {norm:.4f}, tokens/sec = {tokens_per_sec:.2f}")

        lr_scheduler.step()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
