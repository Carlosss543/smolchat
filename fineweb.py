import os
import hashlib
import numpy as np
import tiktoken
from multiprocessing import Pool
from datasets import load_dataset
from tqdm import tqdm


out_dir = os.path.expanduser("~/datasets/edu_fineweb10B")
num_tokens_per_shard = 100_000_000
val_permille = 10 # ~1.0 % of documents go to val

enc = tiktoken.get_encoding("gpt2")


def process(doc):
    doc_id = doc["id"].encode() # every document has a unique ID that we convert in bytes
    h = int(hashlib.md5(doc_id).hexdigest(), 16) % 1000
    split = "val" if h < val_permille else "train"

    tokens = np.array([enc._special_tokens["<|endoftext|>"]] + enc.encode_ordinary(doc["text"]))
    assert (0 <= tokens).all() and (tokens < 2**16).all(), "token dictionary too large for uint16"
    tokens = tokens.astype(np.uint16)

    return split, tokens


class ShardWriter:

    def __init__(self, split):
        self.split = split
        self.buf = np.empty(num_tokens_per_shard, np.uint16)
        self.n = 0
        self.idx = 0

    def add(self, arr):
        while len(arr):
            k = min(len(arr), num_tokens_per_shard - self.n)
            self.buf[self.n:self.n + k] = arr[:k]
            self.n += k
            arr = arr[k:]
            if self.n == num_tokens_per_shard:
                self.flush()

    def flush(self):
        if self.n:
            np.save(f"{out_dir}/{self.split}_{self.idx:06d}.npy", self.buf[:self.n])
            self.idx += 1
            self.n = 0


if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train") # load the dataset in ~/.cache/huggingface
    writers = {"train": ShardWriter("train"), "val": ShardWriter("val")}

    with Pool(max(1, os.cpu_count() // 2)) as pool: # create a pool of worker processes, each of which will process documents in parallel
        for split, toks in tqdm(pool.imap(process, dataset, chunksize=16), total=len(dataset), desc="Tokenizing FineWeb", unit="doc"): # apply the process function to each document in the dataset in parallel
            writers[split].add(toks)

    for w in writers.values(): # flush any remaining tokens in the shard writers
        w.flush()
