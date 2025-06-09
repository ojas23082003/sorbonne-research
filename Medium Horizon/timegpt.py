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
from nixtlats import TimeGPT

timgpt_all_smape = []
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

cols = ["Cases","Total_cases","cases"]
os.environ['TIMEGPT_TOKEN'] = "nixak-ZrbYV8K2HDitCOPxNBqM3QvWp3JiAQJ3I9yWvpkoVMV17o0mrGYxRePT303S20aACSnGISLYwR7lkkUj"
timegpt = TimeGPT(token=os.environ['TIMEGPT_TOKEN'])

for file in excel_files:
    print(f"processing file :{file}")
    df = pd.read_excel(file)
    df['ds'] = pd.to_datetime(df['ds'])

    inferred_freq = pd.infer_freq(df['ds'])
    if inferred_freq is None:
        if "Iquitos" in os.path.basename(file) or "Sanjuan" in os.path.basename(file):
            inferred_freq = 'W'
            continue
        # else:
        #     raise ValueError("Could not infer frequency. Please ensure 'ds' is regularly spaced.")

    for col in cols:
        if col in df.columns:
            col_name = col
            break
    
    series = df[[col_name, 'ds']].copy()
    # series = df[col_name].values
    n_samples = len(series)
    train_len = int(0.95 * n_samples)

    train = series.iloc[:train_len]
    test = series.iloc[train_len:]

    timegpt_fcst = timegpt.forecast(df=train, h=len(test), time_col='ds', target_col=col_name, freq=inferred_freq)
    actuals = test[col_name].tolist()
    predicted = timegpt_fcst['TimeGPT'].tolist()
    all_actuals_combined.append(actuals)
    all_preds_combined.append(predicted)
    timgpt_all_smape.append(smape(actuals,predicted))
    file_names_evaluated.append(file)

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': timgpt_all_smape})
df.to_csv('TimeGPT sMAPE.csv', index=False)

with open('TimeGPT actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_actuals_combined):
        writer.writerow([file_name] + row)

with open('TimeGPT preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_preds_combined):
        writer.writerow([file_name] + row)
    