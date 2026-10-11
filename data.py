import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class TokenDataset(Dataset):

    def __init__(self, path_to_dir, split, sequence_length):
        self.files = sorted(glob.glob(f"{path_to_dir}/{split}_*.npy"))
        self.T = sequence_length
        lens = [np.load(f, mmap_mode="r").shape[0] for f in self.files] # np.load(f, mmap_mode="r") does not load the entire array into memory but only maps it to disk
        n_chunks = [(l - 1) // self.T for l in lens]
        self.cum = np.cumsum([0] + n_chunks)
        self._mm = None

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, idx):
        if self._mm is None:
            self._mm = [np.load(f, mmap_mode="r") for f in self.files]
        s = int(np.searchsorted(self.cum, idx, side="right")) - 1 # index of the shard containing the chunk
        i = (idx - self.cum[s]) * self.T # starting index of the chunk within the shard
        chunk = torch.from_numpy(self._mm[s][i:i + self.T + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def make_loader(path_to_dir, split, batch_size, sequence_length, distributed=False, rank=0, world_size=1, workers=4):
    dataset = TokenDataset(path_to_dir, split, sequence_length)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(split=="train"), drop_last=True) if distributed else None

    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, drop_last=True)

    return data_loader, sampler
