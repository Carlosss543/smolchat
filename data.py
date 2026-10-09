import torch
import tiktoken


class DataLoaderLite:

    def __init__(self, B, T, process_rank, world_size):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        if self.process_rank == 0:
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (self.B * self.T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        self.current_position += B * T * self.world_size # advance the position in the tensor

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.world_size + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y
