import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_size, forecast_size):
        self.series = series
        self.input_size = input_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.series) - self.input_size

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.input_size]
        y = self.series[idx+self.input_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)