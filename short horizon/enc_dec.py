import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=8, num_layers=3, seq_len=10, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        # src shape: (batch, seq_len, input_size)
        # tgt shape: (batch, pred_len, input_size)

        src = self.input_proj(src)  # (batch, seq_len, d_model)
        src = self.pos_encoder(src)

        tgt = self.input_proj(tgt)  # (batch, pred_len, d_model)
        tgt = self.pos_encoder(tgt)

        # Transformer expects (seq_len, batch, d_model)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        output = self.transformer(src, tgt)  # (pred_len, batch, d_model)
        output = self.output_proj(output)  # (pred_len, batch, 1)

        return output.permute(1, 0, 2)  # (batch, pred_len, 1)