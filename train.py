import torch
import argparse
from model import smolTransformer
from data import make_loader
import math
from utils import init_distributed_mode, generate, eval_hellaswag
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from engine import train_one_step, evaluate
import tiktoken
from datasets import load_dataset
import wandb
import os


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--sequence_length", type=int, default=1024)

    parser.add_argument("--global_batch_size", type=int, default=2**19) # 2**19 ~500k tokens per batch
    parser.add_argument("--micro_batch_size", type=int, default=16)

    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=200)

    parser.add_argument("--weight_decay", type=float, default=0.1)

    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--wandb_run_id", type=str, default=None)

    return parser


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a compatible NVIDIA GPU."


    # --- reproducibility ---
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    # --- distributed training setup ---
    init_distributed_mode(args)


    # --- load data ---
    train_loader, _ = make_loader(path_to_dir=args.data_dir, split="train", batch_size=args.micro_batch_size, sequence_length=args.sequence_length, distributed=args.distributed, rank=args.ddp_rank, world_size=args.ddp_world_size)
    val_loader, _ = make_loader(path_to_dir=args.data_dir, split="val", batch_size=args.micro_batch_size, sequence_length=args.sequence_length, distributed=args.distributed, rank=args.ddp_rank, world_size=args.ddp_world_size)

    tokenizer = tiktoken.get_encoding("gpt2")

    hs_examples = list(load_dataset("Rowan/hellaswag", split="validation"))


    # --- initialize model ---
    base_model = smolTransformer(sequence_length=args.sequence_length).to(args.device)

    model = torch.compile(base_model)

    if args.distributed:
        model = DDP(model, device_ids=[args.ddp_local_rank])

    args.num_model_parameters = sum(p.numel() for p in model.parameters())


    # --- optimizer and gradient scaler for fp16 ---
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2] # weights and embeddings 
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2] # biases and layernorms

    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': args.weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True
    )

    scaler = torch.amp.GradScaler("cuda")


    # --- compute number of training steps ---
    assert args.global_batch_size % (args.micro_batch_size * args.sequence_length * args.ddp_world_size) == 0, "Global batch size must be divisible by micro batch size * sequence length * number of distributed processes"
    args.grad_accum_steps = args.global_batch_size // (args.micro_batch_size * args.sequence_length * args.ddp_world_size)

    available_steps = len(train_loader) // args.grad_accum_steps
    args.max_steps = min(args.max_steps, available_steps) if args.max_steps is not None else available_steps


    # --- learning rate scheduler with warmup and cosine decay ---
    def lr_lambda(step):
        max_lr = args.lr
        min_lr = max_lr * 0.1

        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps

        elif step >= args.max_steps:
            return min_lr / max_lr

        else:
            decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (min_lr + coeff * (max_lr - min_lr)) / max_lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    # --- resume optimizer, lr_scheduler, and scaler ---
    starting_step = 0

    if args.resume_from_checkpoint:
        assert args.wandb_run_id is not None, "wandb_run_id must be specified when resuming from checkpoint."
        assert os.path.exists(f"{args.output_dir}/{args.wandb_run_id}.pt"), f"Checkpoint path '{args.output_dir}/{args.wandb_run_id}.pt' does not exist."

        checkpoint = torch.load(f"{args.output_dir}/{args.wandb_run_id}.pt", map_location=args.device)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        starting_step = checkpoint["step"] + 1
        del checkpoint # free up memory

        if args.master_process:
            print(f"Resumed training from checkpoint: {args.output_dir}/{args.wandb_run_id}.pt")


    # --- initialize wandb ---
    if args.master_process:
        config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        run = wandb.init(
            project="smolchat",
            dir="./wandb_logs",
            config=config,
            mode="online",
            id=args.wandb_run_id if args.resume_from_checkpoint else None,
            resume="must" if args.resume_from_checkpoint else False
        )
        wandb.define_metric("step")
        for metric in ("train_loss", "val_loss", "hellaswag_acc", "lr", "grad_norm", "tokens_per_sec"):
            wandb.define_metric(metric, step_metric="step")


    # --- training loop (we do at most one epoch) ---
    train_iter = iter(train_loader)

    batches_to_skip = starting_step * args.grad_accum_steps
    for _ in range(batches_to_skip): # skip batches if resuming from a checkpoint
        next(train_iter)

    for step in range(starting_step, args.max_steps):
        train_one_step(args, model, train_iter, optimizer, scaler, step)
        lr_scheduler.step()

        if (step+1) % 200 == 0 or step == args.max_steps - 1:
            evaluate(args, model, val_loader, step, val_steps=1)
            generate(args, model, tokenizer, "France is", num_examples=1, max_new_tokens=30, temperature=1.0)
            eval_hellaswag(args, model, tokenizer, hs_examples, step)

        if ((step+1) % 500 == 0 or step == args.max_steps - 1) and args.master_process:
            model_to_save = model.module if hasattr(model, "module") else model # Unwrap the model if it's wrapped by DDP
            model_to_save = model_to_save._orig_mod if hasattr(model_to_save, "_orig_mod") else model_to_save # Unwrap the model if it's wrapped by torch.compile
            checkpoint = {
                "step": step,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": config
            }
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(checkpoint, f"{args.output_dir}/{run.id}.pt")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
