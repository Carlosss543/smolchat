import os
import torch
from torch.distributed import init_process_group


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
