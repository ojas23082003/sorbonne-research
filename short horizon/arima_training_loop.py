import os
import csv
import glob
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
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

        # Use train+val for fitting, test for evaluation
        train_series = series[:train_len + val_len]
        test_series = series[train_len + val_len:]

        # ARIMA order selection (simple grid, can be improved)
        best_aic = float('inf')
        best_order = None
        # for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(train_series, order=(10, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (10, d, q)
                except Exception:
                    continue

        # Fit best ARIMA model
        model = ARIMA(train_series, order=best_order)
        model_fit = model.fit()

        # Rolling forecast
        preds = []
        history = list(train_series)
        for t in range(len(test_series)):
            forecast = model_fit.forecast(steps=1)[0]
            preds.append(forecast)
            history.append(test_series[t])
            # Refit model for next step (optional, slow for large test sets)
            # model_fit = ARIMA(history, order=best_order).fit()

        all_preds_combined.append(np.array(preds))
        all_actuals_combined.append(test_series)

        test_smape = smape(test_series, np.array(preds))
        lstm_all_smape.append(test_smape)
        file_names_evaluated.append(file)

    except Exception as e:
        print(f"the error occurred in {file}: {e}")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': lstm_all_smape})
df.to_csv('ARIMA_sMAPE.csv', index=False)

with open('arima actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_actuals_combined):
        writer.writerow([file_name] + row.flatten().tolist())

with open('arima preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_preds_combined):
        writer.writerow([file_name] + row.flatten().tolist())