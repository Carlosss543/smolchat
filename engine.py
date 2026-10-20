import torch
import torch.distributed as dist
from contextlib import nullcontext
import time
import wandb


def train_one_step(model, train_iter, optimizer, scaler, args, grad_accum_steps, step):
    model.train()
    t0 = time.time()
    loss_accum = torch.tensor(0.0, device=args.device)

    for micro_step in range(grad_accum_steps):
        x, y = next(train_iter)
        x, y = x.to(args.device), y.to(args.device)

        sync_context = (model.no_sync() if args.distributed and (micro_step+1) < grad_accum_steps else nullcontext())

        with sync_context:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)

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
        wandb.log({"step": step, "train_loss": loss_accum.item(), "lr": optimizer.param_groups[0]['lr'], "grad_norm": norm, "tokens_per_sec": tokens_per_sec})
        print(f"Step {step}: loss = {loss_accum:.4f}, lr = {optimizer.param_groups[0]['lr']:.4e}, grad_norm = {norm:.4f}, tokens/sec = {tokens_per_sec:.2f}")


@torch.no_grad()
def evaluate(model, loader, args, grad_accum_steps, step, val_steps):
    model.eval()
    val_iter = iter(loader)
    val_loss_accum = torch.tensor(0.0, device=args.device)

    for _ in range(val_steps * grad_accum_steps):
        x, y = next(val_iter)
        x, y = x.to(args.device), y.to(args.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, y)
        loss = loss / (val_steps * grad_accum_steps)
        val_loss_accum += loss.detach()

    if args.distributed:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    if args.master_process:
        wandb.log({"step": step, "val_loss": val_loss_accum.item()})
        print(f"Validation loss = {val_loss_accum:.4f}")

    return val_loss_accum.item()
