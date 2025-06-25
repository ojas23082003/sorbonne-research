import os
import csv
import glob
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from smape import smape

lstm_all_smape = []
file_names_evaluated = []
all_preds_combined = []
all_actuals_combined = []
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting', 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

cols = ["Cases", "Total_cases", "cases"]
input_size = 10

def create_sequences(series, input_size):
    X, y = [], []
    for i in range(len(series) - input_size):
        X.append(series[i:i+input_size])
        y.append(series[i+input_size])
    return np.array(X), np.array(y)

for file in excel_files:
    try:
        df = pd.read_excel(file)
        col_name = None
        for col in cols:
            if col in df.columns:
                col_name = col
                break
        if col_name is None:
            print(f"No valid column found in {file}")
            continue

        series = df[col_name].values.astype(float)
        n_samples = len(series)
        train_len = int(0.90 * n_samples)
        val_len = int(0.08 * n_samples)

        # Prepare train, val, test splits
        train_series = series[:train_len]
        val_series = series[train_len:train_len+val_len]
        test_series = series[train_len+val_len:]

        # Prepare sequences
        X_train, y_train = create_sequences(train_series, input_size)
        X_val, y_val = create_sequences(val_series, input_size)
        # For test, use the last input_size values from train+val as context
        context = np.concatenate([train_series, val_series])[-input_size:]
        X_test = []
        for i in range(len(test_series)):
            X_test.append(context[-input_size:])
            context = np.append(context, test_series[i])
        X_test = np.array(X_test)

        # Hyperparameters
        n_estimators_list = [50, 100]
        max_depth_list = [3, 5]
        best_val_loss = float('inf')
        best_config = None
        best_model = None

        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, objective='reg:squarederror')
                model.fit(X_train, y_train)
                val_preds = model.predict(X_val)
                val_loss = mean_squared_error(y_val, val_preds)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = {'n_estimators': n_estimators, 'max_depth': max_depth}
                    best_model = model

        # Retrain on train+val for final prediction
        full_train_series = np.concatenate([train_series, val_series])
        X_full, y_full = create_sequences(full_train_series, input_size)
        final_model = XGBRegressor(
            n_estimators=best_config['n_estimators'],
            max_depth=best_config['max_depth'],
            objective='reg:squarederror'
        )
        final_model.fit(X_full, y_full)

        # Predict on test set
        preds = []
        context = full_train_series[-input_size:].tolist()
        for i in range(len(test_series)):
            x_input = np.array(context[-input_size:]).reshape(1, -1)
            pred = final_model.predict(x_input)[0]
            preds.append(pred)
            context.append(pred)  # Use actual value for next input

        all_preds_combined.append(np.array(preds))
        all_actuals_combined.append(test_series)

        test_smape = smape(test_series, np.array(preds))
        lstm_all_smape.append(test_smape)
        file_names_evaluated.append(file)

    except Exception as e:
        print(f"the error occurred in {file}: {e}")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': lstm_all_smape})
df.to_csv('XGBoost_sMAPE.csv', index=False)

with open('xgboost actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_actuals_combined):
        writer.writerow([file_name] + row.flatten().tolist())

with open('xgboost preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_preds_combined):
        writer.writerow([file_name] + row.flatten().tolist())