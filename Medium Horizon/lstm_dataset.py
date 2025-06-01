import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len  # avoid index out of range

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]  # shape: (seq_len, 1)
        y = self.data[idx + self.seq_len]

        # x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        # y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        return x, y