import os
import glob
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
    try:
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
        test_dataset = TimeSeriesDataset(test_series_scaled, input_size, forecast_size)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
        criterion = nn.MSELoss()

        print("\n")
        print("\n")
        print("\n")
        print(f"############ TRAINING STARTED FOR FILE {file} ############")
        print("\n")
        print("\n")
        print("\n")
        for epoch in range(50): 
            train_epoch(model, full_train_loader, optimizer, criterion)

        model.eval()
        all_smape = []
        all_preds = []
        all_actuals = []

        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)

                preds_np = preds.cpu().numpy().flatten().reshape(-1, 1)
                yb_np = yb.cpu().numpy().flatten().reshape(-1, 1)

                preds_orig = scaler.inverse_transform(preds_np).flatten()
                yb_orig = scaler.inverse_transform(yb_np).flatten()

                batch_smape = smape(yb_orig, preds_orig)
                all_smape.append(batch_smape.item())

                # Collect predictions and actuals
                all_preds.extend(preds.cpu().numpy().flatten())
                all_actuals.extend(yb.cpu().numpy().flatten())
        
        print("\n")
        print("\n")
        print("\n")
        print(f"############ EVALUATION ENDED FOR FILE {file} ############")
        print("\n")
        print("\n")
        print("\n")
        test_smape = sum(all_smape) / len(all_smape)
        nbeats_all_smape.append(test_smape)
        file_names_evaluated.append(file)
        print(f"---------------------------- FINAL SMAPE: {test_smape} ----------------------------")
    except:
        print(f"error found in {file}. Continuing to the next...")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': nbeats_all_smape})
df.to_csv('NBEATS sMAPE.csv', index=False)