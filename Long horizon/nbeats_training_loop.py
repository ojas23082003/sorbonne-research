import os
import csv
import glob
import time
import optuna
import numpy as np
import pandas as pd
from nbeats import *
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from smape import smape


nbeats_all_smape = []
file_names_evaluated = []
time_to_train = []
time_to_infer = []
all_predictions_combined = []
all_actuals_combined = []
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()   
        forecast = model(x)
        loss = criterion(forecast, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            forecast = model(x)
            loss = criterion(forecast, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def objective(trial):
    # Fixed input and forecast size
    input_size = 10
    forecast_size = 1

    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 10, 30)
    n_blocks = trial.suggest_int('n_blocks', 2, 8)
    n_layers = trial.suggest_int('n_layers', 2, 6)
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    n_samples = len(series)
    train_len = int(0.75 * n_samples)
    val_len = int(0.15 * n_samples)

    scaler = StandardScaler()
    train_series = series[:train_len]
    train_series_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

    val_series = series[train_len:train_len+val_len]
    val_series_scaled = scaler.transform(val_series.reshape(-1, 1)).flatten()

    train_dataset = TimeSeriesDataset(train_series_scaled, input_size, forecast_size)
    val_dataset = TimeSeriesDataset(val_series_scaled, input_size, forecast_size)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Define model
    model = NBeats(input_size=input_size, forecast_size=forecast_size,
                   hidden_size=hidden_size, n_blocks=n_blocks, n_layers=n_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train for a few epochs
    for epoch in range(20):
        train_epoch(model, train_loader, optimizer, criterion)

    # Validate
    val_loss = evaluate(model, val_loader, criterion)
    return val_loss

cols = ["Cases","Total_cases","cases"]

for file in excel_files:
    # try:
    df = pd.read_excel(file)

    for col in cols:
        if col in df.columns:
            col_name = col

    series = df[col_name].values  # replace 'Value' with your actual column name

    input_size = 10
    forecast_size = 1
    dataset = TimeSeriesDataset(series, input_size, forecast_size)

    # Step 4: Wrap with DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # print(study.best_trial.params)
    best_params = study.best_trial.params
    print("\n")
    print("\n")
    print("\n")
    print(f"############ FINISHED EVALUATION FOR FILE {file} ############")
    print("\n")
    print("\n")
    print("\n")

    model = NBeats(
        input_size=10,
        forecast_size=1,
        hidden_size=best_params['hidden_size'],
        n_blocks=best_params['n_blocks'],
        n_layers=best_params['n_layers']
    )

    input_size = 10
    forecast_size = 1

    scaler = StandardScaler()

    n_samples = len(series)
    train_len = int(0.75 * n_samples)
    val_len = int(0.15 * n_samples)

    # making full train data
    full_train_series = series[:train_len+val_len]
    full_train_series_scaled = scaler.fit_transform(full_train_series.reshape(-1, 1)).flatten()
    train_dataset = TimeSeriesDataset(full_train_series_scaled, input_size, forecast_size)
    full_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_series = series[train_len+val_len:]
    test_series_scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()
    # test_dataset = TimeSeriesDataset(test_series_scaled, input_size, forecast_size)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    


    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    print("\n")
    print("\n")
    print("\n")
    print(f"############ TRAINING STARTED FOR FILE {file} ############")
    print("\n")
    print("\n")
    print("\n")
    time_to_train_start = time.time()
    for epoch in range(50): 
        train_epoch(model, full_train_loader, optimizer, criterion)

    time_to_train_end = time.time()
    time_to_train.append(time_to_train_end-time_to_train_start)
    model.eval()
    all_smape = []
    all_preds = []
    all_actuals = []

    # time_to_infer_start = time.time()

    current_input = full_train_series_scaled[-input_size:].tolist()

    with torch.no_grad():
        for i in range(len(test_series_scaled)):
            input_seq = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0)

            forecast = model(input_seq)
            forecast_np = forecast.cpu().numpy().flatten()[0]
            all_preds.append(forecast_np)
            
            current_input.pop(0)
            current_input.append(forecast_np)
    
    all_preds = np.array(all_preds).reshape(-1,1)
    all_actuals = test_series_scaled
    all_actuals = np.array(all_actuals).reshape(-1,1)

    pred_orig = scaler.inverse_transform(all_preds).flatten()
    actuals_orig = scaler.inverse_transform(all_actuals).flatten()
    
    # time_to_infer_end = time.time()
    # time_to_infer.append(time_to_infer_end-time_to_infer_start)

    all_predictions_combined.append(pred_orig)
    all_actuals_combined.append(actuals_orig)
    
    print("\n")
    print("\n")
    print("\n")
    print(f"############ EVALUATION ENDED FOR FILE {file} ############")
    print("\n")
    print("\n")
    print("\n")
    test_smape = smape(actuals_orig,pred_orig)
    nbeats_all_smape.append(test_smape)
    file_names_evaluated.append(file)
    print(f"---------------------------- FINAL SMAPE: {test_smape} ----------------------------")
    # except:
    #     print(f"error found in {file}. Continuing to the next...")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': nbeats_all_smape})
df.to_csv('NBEATS sMAPE.csv', index=False)

# df = pd.DataFrame(all_actuals_combined)
with open('nbeats actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_actuals_combined):
        writer.writerow([file_name] + row.flatten().tolist())

with open('nbeats preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_predictions_combined):
        writer.writerow([file_name] + row.flatten().tolist())