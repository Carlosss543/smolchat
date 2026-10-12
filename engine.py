import torch
import torch.distributed as dist
from contextlib import nullcontext
import time


def train_one_step(model, train_iter, optimizer, scaler, criterion_train, args, grad_accum_steps, step):
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


@torch.no_grad()
def evaluate(model, loader, criterion, args, grad_accum_steps, val_steps):
    model.eval()
    val_iter = iter(loader)
    val_loss_accum = torch.tensor(0.0, device=args.device)

    for _ in range(val_steps * grad_accum_steps):
        x, y = next(val_iter)
        x, y = x.to(args.device), y.to(args.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / (val_steps * grad_accum_steps)
        val_loss_accum += loss.detach()

    if args.distributed:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    if args.master_process:
        print(f"Validation loss = {val_loss_accum:.4f}")

    return val_loss_accum.item()


@torch.no_grad()
def generate(model, enc, prompt, args, num_examples=1, max_new_tokens=100, temperature=1.0, top_k=50):
    model.eval()
    ids = torch.tensor([enc.encode_ordinary(prompt)], device=args.device).repeat(num_examples, 1)

    sample_rng = torch.Generator(device=args.device)
    sample_rng.manual_seed(42+args.ddp_rank)

    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(ids[:, -args.sequence_length:]) # (B, T, vocab_size)
        logits = logits[:, -1, :] / temperature # (B, vocab_size)
        probs = logits.softmax(dim=-1)

        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1) # (B, k), (B, k)
        choice = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # (B, 1), indices in [0, k[
        next_id = torch.gather(topk_idx, -1, choice) # (B, 1), vrai id de token

        ids = torch.cat([ids, next_id], dim=1)

    print([enc.decode(seq.tolist()) for seq in ids])

    return [enc.decode(seq.tolist()) for seq in ids]
