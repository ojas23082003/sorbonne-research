import os
import csv
import glob
import optuna
import numpy as np
import pandas as pd
from lstm import *
from lstm_dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from smape import smape
import torch.optim as optim

lstm_all_smape = []
file_names_evaluated = []
all_preds_combined = []
all_actuals_combined = []
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

def train_and_evaluate(model, train_loader, val_loader, num_epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:

            optimizer.zero_grad()
            output = model(x_batch)

            # Since y_batch is (batch, out_len, 1) and output is (batch, out_len, 1)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            # x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * x_batch.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    return avg_val_loss

cols = ["Cases","Total_cases","cases"]
input_size = 10

for file in excel_files:
    
    # try:
    df = pd.read_excel(file)
    for col in cols:
        if col in df.columns:
            col_name = col

    series = df[col_name].values

    scaler = StandardScaler()
    n_samples = len(series)
    train_len = int(0.75 * n_samples)
    val_len = int(0.15 * n_samples)

    train_series = series[:train_len]
    train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    train_dataset = TimeSeriesDataset(train_series_scaled, 10)
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

    val_series = series[train_len:train_len+val_len]
    val_series_scaled = scaler.transform(val_series.reshape(-1,1)).flatten()
    val_dataset = TimeSeriesDataset(val_series_scaled,10)
    val_loader = DataLoader(val_dataset,batch_size=32)

    hidden_sizes = [32, 64, 128]
    learning_rates = [0.001, 0.0005]
    num_epochs = 50

    best_val_loss = float('inf')
    best_config = None

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            print(f"Training with hidden_size={hidden_size}, lr={lr}")

            model = CustomLSTMForecast(
                input_size=1,
                hidden_size=hidden_size,
                output_size=1, 
                seq_len=10
            )

            val_loss = train_and_evaluate(model, train_loader, val_loader, num_epochs, lr)

            print(f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = {
                    "hidden_size": hidden_size,
                    "learning_rate": lr
                }
    
    full_train_series = series[:train_len+val_len]
    full_train_scaled = scaler.fit_transform(full_train_series.reshape(-1,1)).flatten()
    final_dataset = TimeSeriesDataset(full_train_scaled, 10)
    final_loader  = DataLoader(final_dataset, batch_size=32, shuffle=True)

    final_model = CustomLSTMForecast(
        input_size=1,
        hidden_size=best_config['hidden_size'],
        output_size=1,  # one-step forecasting
        seq_len=10
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config['learning_rate'])

    num_epochs = 100

    for epoch in range(num_epochs):
        final_model.train()
        for x_batch, y_batch in final_loader:

            optimizer.zero_grad()
            output = final_model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # (Optional: log training loss every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    test_series  = series[train_len+val_len:]
    test_scaled = scaler.transform(test_series.reshape(-1,1)).flatten()
    # test_dataset = TimeSeriesDataset(test_scaled,10)
    # test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    final_model.eval()

    all_predictions = []
    all_ground_truths = test_scaled
    current_input = full_train_scaled[-input_size:].tolist()

    with torch.no_grad():
        for i in range(len(test_scaled)):

            input_seq = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            forecast = final_model(input_seq)
            forecast = forecast.flatten()
            all_predictions.append(forecast[0])

            current_input.pop(0)
            current_input.append(forecast[0])


    # Convert to numpy arrays
    # all_predictions = np.array(all_predictions)
    # all_ground_truths = np.array(all_ground_truths)

    all_predictions = np.array(all_predictions).reshape(-1, 1)
    all_ground_truths = np.array(all_ground_truths).reshape(-1, 1)


    all_predictions_original = scaler.inverse_transform(all_predictions).flatten()
    all_ground_truths_original = scaler.inverse_transform(all_ground_truths).flatten()

    all_preds_combined.append(all_predictions_original)
    all_actuals_combined.append(all_ground_truths_original)

    test_smape = smape(all_ground_truths_original, all_predictions_original)

    lstm_all_smape.append(test_smape)
    file_names_evaluated.append(file)

    # except Exception as e:
    #     print(f"the error occured is {e}")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': lstm_all_smape})
df.to_csv('LSTM sMAPE.csv', index=False)

with open('lstm actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_actuals_combined):
        writer.writerow([file_name] + row.flatten().tolist())

with open('lstm preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_preds_combined):
        writer.writerow([file_name] + row.flatten().tolist())