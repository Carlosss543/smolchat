import torch
import argparse
from model import smolTransformer
from data import make_loader
import math
from utils import init_distributed_mode
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from engine import train_one_step, evaluate, generate
import tiktoken


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
    val_loader, _ = make_loader(path_to_dir=args.data_dir, split="val", batch_size=args.micro_batch_size, sequence_length=args.sequence_length, distributed=args.distributed, rank=args.ddp_rank, world_size=args.ddp_world_size)

    tokenizer = tiktoken.get_encoding("gpt2")


    # --- compute number of training steps ---
    assert args.global_batch_size % (args.micro_batch_size * args.sequence_length * args.ddp_world_size) == 0, "Global batch size must be divisible by micro batch size * sequence length * number of distributed processes"
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * args.sequence_length * args.ddp_world_size)

    available_steps = len(train_loader) // grad_accum_steps
    max_steps = min(args.max_steps, available_steps) if args.max_steps is not None else available_steps

    if args.master_process:
        print(f"max_steps = {max_steps}, grad_accum_steps = {grad_accum_steps}")


    # --- initialize model ---
    model = smolTransformer(sequence_length=args.sequence_length).to(args.device)

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
    criterion_val = torch.nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda")


    # --- learning rate scheduler with warmup and cosine decay ---
    def lr_lambda(step):
        max_lr = args.lr
        min_lr = max_lr * 0.1

        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps

        elif step >= max_steps:
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
        train_one_step(model, train_iter, optimizer, scaler, criterion_train, args, grad_accum_steps, step)

        lr_scheduler.step()

        if step % 100 == 0:
            evaluate(model, val_loader, criterion_val, args, grad_accum_steps, val_steps=1)
            generate(model, tokenizer, "France is", args, num_examples=1, max_new_tokens=30, temperature=1.0)
            model.train()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
