import os
import csv
import glob
import torch
import optuna
import numpy as np
import pandas as pd
from smape import smape
import torch.optim as optim
from chronos import ChronosPipeline
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

chronos_all_smape = []
file_names_evaluated = []
all_preds_combined = []
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        )

cols = ["Cases","Total_cases","cases"]

for file in excel_files:
    try:
        df = pd.read_excel(file)

        # Step 2: Extract the time series column
        for col in cols:
            if col in df.columns:
                col_name = col
                break
        series = df[col_name].values  # replace 'Value' with your actual column name
        input_size = 10
        forecast_size = 1

        n_samples = len(series)
        train_len = int(0.75 * n_samples)
        val_len = int(0.15 * n_samples)

        context = torch.tensor(series[:train_len+val_len])
        prediction_length = n_samples - train_len - val_len
        forecast = pipeline.predict(context, prediction_length) 

        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        all_actuals = series[-prediction_length:]

        all_preds_combined.append(median)

        pred_smape = smape(all_actuals,median)
        chronos_all_smape.append(pred_smape)
        file_names_evaluated.append(file)
    except Exception as e:
        print(f"{e}")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': chronos_all_smape})
df.to_csv('Chronos t5 mini sMAPE.csv', index=False)

with open('chronos mini preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_preds_combined):
        writer.writerow([file_name] + row.flatten().tolist())