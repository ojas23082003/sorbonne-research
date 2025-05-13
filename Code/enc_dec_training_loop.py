import os
import glob
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from enc_dec import *
from smape import smape
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

enc_dec_all_smape = []
file_names_evaluated = []
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        context, target = self.sequences[idx]
        context = torch.tensor(context, dtype=torch.float32).unsqueeze(-1)  # shape: (seq_len, 1)
        target = torch.tensor([target], dtype=torch.float32)  # shape: (1,)
        return context, target

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        context = data[i:i+seq_len]
        target = data[i+seq_len]
        sequences.append((context, target))
    return sequences

cols = ["Cases","Total_cases","cases"]

for file in excel_files:
    df = pd.read_excel(file)

    for col in cols:
        if col in df.columns:
            col_name = col
    
    series = df[col_name].values.reshape(-1, 1)  # Reshape for scaler

    # Split train/test before sequence creation (to avoid leakage)
    train_size = int(0.9 * len(series))
    train_series = series[:train_size]
    test_series = series[train_size:]

    if len(test_series) < 11:  # 10 context + 1 target
        needed = 11 - len(test_series)
        # Move needed elements from train to test
        test_series = np.concatenate((train_series[-needed:], test_series))
        train_series = train_series[:-needed]

    # Fit scaler only on train data
    scaler = StandardScaler()
    scaler.fit(train_series)

    # Scale train and test series
    train_series_scaled = scaler.transform(train_series).flatten()
    test_series_scaled = scaler.transform(test_series).flatten()

    train_seq = create_sequences(train_series_scaled, seq_len=10)
    test_seq = create_sequences(test_series_scaled, seq_len=10)

    train_dataset = TimeSeriesDataset(train_seq)
    test_dataset = TimeSeriesDataset(test_seq)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = TimeSeriesTransformer()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()

            # Prepare target input for decoder (during training, teacher forcing)
            tgt_input = context[:, -1:, :]  # last timestep as decoder input

            output = model(context, tgt_input)  # (batch, 1, 1)
            loss = criterion(output.squeeze(), target.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    model.eval()
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for context, target in test_loader:
            # Prepare decoder input (last value of context)
            tgt_input = context[:, -1:, :]  # shape: (batch, 1, 1)

            # Get model prediction
            output = model(context, tgt_input)  # (batch, 1, 1)
            preds = output.squeeze(-1).squeeze(-1)  # shape: (batch,)

            # Collect predictions and ground truth
            all_predictions.extend(preds.numpy())
            all_ground_truth.extend(target.squeeze(-1).numpy())

    all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
    all_ground_truth = scaler.inverse_transform(np.array(all_ground_truth).reshape(-1, 1)).flatten()

    enc_dec_all_smape.append(smape(all_ground_truth,all_predictions))
    file_names_evaluated.append(file)

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': enc_dec_all_smape})
df.to_csv('Enc Dec sMAPE.csv', index=False)