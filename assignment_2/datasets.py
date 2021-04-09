import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Iterable

def read_file(filename: str, chunk_size=4000) -> Iterable[int]:
    with open(filename, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            yield from b
            if len(b) < chunk_size:
                return

class SerialCharData(Dataset):
    def __init__(self, snippet_length: int, source_filename: str):
        self.snippet_length = snippet_length
        self.data = np.fromiter(read_file(source_filename), np.ubyte)
        self.wrap = np.concatenate((self.data[-(self.snippet_length + 1):],
                                    self.data[:self.snippet_length + 1]))
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        use_data, use_index = ((self.data, index)
                               if index + self.snippet_length + 1 < len(self.data)
                               else (self.wrap, index + self.sippet_length + 1 - len(self.data)))
        return (torch.tensor(self.data[index : index + self.snippet_length]),
                torch.tensor(self.data[index + 1 : index + self.snippet_length + 1]))
