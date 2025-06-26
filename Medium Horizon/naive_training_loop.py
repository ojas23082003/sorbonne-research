import os
import csv
import glob
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NaiveSeasonal
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

# Containers
all_combined_preds = []
all_combined_actuals = []
all_filenames = []

# Setup file path
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.abspath(os.path.join(current_dir, '..', 'Epicasting', 'Merged'))

# Excel file extensions and column candidates
extensions = ('*.xlsx', '*.xls')
cols = ["Cases", "Total_cases", "cases"]

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

# NaiveSeasonal parameter: season length
K = 3

# Iterate over all Excel files
for file in excel_files:
    try:
        df = pd.read_excel(file)

        # Identify time series column
        for col in cols:
            if col in df.columns:
                col_name = col
                break

        series = TimeSeries.from_dataframe(df, value_cols=col_name, freq="D")

        # Scale the data
        scaler = Scaler(StandardScaler())
        series = scaler.fit_transform(series)

        # Train-validation split
        train, val = series.split_before(0.95)

        # Create and fit NaiveSeasonal model
        model = NaiveSeasonal(K=K)
        model.fit(train)

        print(f"Naive prediction for {file}")

        # Predict the full validation horizon in one go
        forecast = model.predict(n=len(val))

        # Inverse transform predictions and actuals
        forecast = scaler.inverse_transform(forecast)
        val = scaler.inverse_transform(val)

        preds_df = forecast.to_dataframe()
        preds_list = preds_df[col_name].tolist()
        all_combined_preds.append(preds_list)

        actuals_df = val.to_dataframe()
        actuals_list = actuals_df[col_name].tolist()
        all_combined_actuals.append(actuals_list)

        all_filenames.append(file)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Save predictions and actuals
with open('naive actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(all_filenames, all_combined_actuals):
        writer.writerow([file_name] + row)

with open('naive preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(all_filenames, all_combined_preds):
        writer.writerow([file_name] + row)
