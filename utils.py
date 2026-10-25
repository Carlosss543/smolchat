import os
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group
import torch.distributed as dist
import wandb


def init_distributed_mode(args):
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        args.ddp_rank = int(os.environ["RANK"])
        args.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        args.ddp_world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = True
        args.master_process = (args.ddp_rank == 0)
        args.device = torch.device(f"cuda:{args.ddp_local_rank}")
        torch.cuda.set_device(args.device) # for security, set the current CUDA device to the one specified in args.device, .to("cuda") becomes equivalent to .to(args.device)
    else:
        args.ddp_rank = 0
        args.ddp_local_rank = 0
        args.ddp_world_size = 1
        args.distributed = False
        args.master_process = True
        args.device = torch.device("cuda")


@torch.no_grad()
def generate(args, model, enc, prompt, num_examples=1, max_new_tokens=100, temperature=1.0, top_k=50):
    model.eval()
    ids = torch.tensor([enc.encode_ordinary(prompt)], device=args.device).repeat(num_examples, 1)

    sample_rng = torch.Generator(device=args.device)
    sample_rng.manual_seed(42+args.ddp_rank)

    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(ids[:, -args.sequence_length:]) # (B, T, vocab_size)
        logits = logits[:, -1, :] / temperature # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1) # (B, k), (B, k)
        choice = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # (B, 1), indices in [0, k[
        next_id = torch.gather(topk_idx, -1, choice) # (B, 1), vrai id de token

        ids = torch.cat([ids, next_id], dim=1)

    print([enc.decode(seq.tolist()) for seq in ids])

    return [enc.decode(seq.tolist()) for seq in ids]


@torch.no_grad()
def eval_hellaswag(args, model, tokenizer, examples, step, num_examples_used=None):
    model.eval()

    correct = torch.tensor(0, device=args.device)
    total = torch.tensor(0, device=args.device)

    N = len(examples) if num_examples_used is None else min(num_examples_used, len(examples))

    # every GPU only processes the examples corresponding to its rank (stride = world_size)
    for idx in range(args.ddp_rank, N, args.ddp_world_size):
        example = examples[idx]

        # encode the context and each of the possible endings using the tokenizer
        ctx = tokenizer.encode(example["ctx"])
        seqs, masks = [], []
        for end in example["endings"]:
            e = tokenizer.encode(" " + end) # the empty space before the ending is important with GPT-2 BPE
            seqs.append(ctx + e)
            masks.append([0] * len(ctx) + [1] * len(e))

        # pad sequences to the same length
        T = max(len(s) for s in seqs)
        tok = torch.zeros(4, T, dtype=torch.long)
        mask = torch.zeros(4, T, dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            tok[i, :len(s)] = torch.tensor(s)
            mask[i, :len(m)] = torch.tensor(m)
        tok, mask = tok.to(args.device), mask.to(args.device)

        logits = model(tok) # (4, L, vocab_size)

        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        shift_tok = tok[:, 1:].reshape(-1)
        shift_mask = mask[:, 1:]
        loss = F.cross_entropy(shift_logits, shift_tok, reduction="none").view(4, -1) # (4, T-1) loss for each token in each sequence
        avg = (loss * shift_mask).sum(1) / shift_mask.sum(1)   # average loss over the ending only

        correct += (avg.argmin().item() == int(example["label"]))
        total += 1

    if args.distributed:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    acc = correct / total if total > 0 else 0.0

    if args.master_process:
        wandb.log({"step": step, "hellaswag_acc": acc})
        print(f"HellaSwag accuracy = {acc:.4f}")

    return acc
