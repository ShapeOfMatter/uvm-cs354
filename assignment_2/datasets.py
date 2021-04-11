import numpy as np
import random  # I think this is good enough.
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Iterable

def read_file(filename: str, chunk_size=4000) -> Iterable[int]:
    with open(filename, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            yield from b
            if len(b) < chunk_size:
                return

class SerialCharData(Dataset):
    def __init__(self, source_filename: str, snippet_length: int):
        self.snippet_length = snippet_length
        self.data = np.fromiter(read_file(source_filename), np.ubyte)
        self.wrap = np.concatenate((self.data[-(self.snippet_length + 1):],
                                    self.data[:self.snippet_length + 1]))
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        use_data, use_index = ((self.data, index)
                               if index + self.snippet_length + 1 < len(self.data)
                               else (self.wrap, index + self.snippet_length + 1 - len(self.data)))
        def as_torch_ints(xs, start, length):  # transforming to ints may be pointless...
            return torch.tensor([int(x) for x in xs[start : start + length]])
        return (as_torch_ints(use_data, use_index, self.snippet_length),
                as_torch_ints(use_data, use_index + 1, self.snippet_length))

    def get_dataloader(self, batch_size: int, book_size: int):
        return DataSubSet(self, book_size, random.randrange(len(self))).as_dataloader(batch_size)

class DataSubSet(Dataset):
    def __init__(self, source: SerialCharData, size: int, offset: int):
        self.source = source
        self.source_size = len(source)
        self.size = size
        self.offset = offset
        
    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.source.__getitem__((self.offset + index) % self.source_size)

    def as_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, shuffle=True)


