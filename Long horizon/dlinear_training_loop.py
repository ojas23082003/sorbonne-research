import os
import csv
import glob
import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from darts.models import DLinearModel
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

# DLinear model parameters
input_chunk_length = 3
output_chunk_length = 1

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
        train, val = series.split_before(0.90)

        # Create DLinear model
        model = DLinearModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            random_state=42,
            n_epochs=30,
            batch_size=32,
        )

        model.fit(train, verbose=False)

        print(f"Rolling window prediction for {file}")
        context_window = train.copy()
        context_window = context_window[-input_chunk_length:]

        rolling_preds = []
        n_forecasts = len(val)

        for i in range(n_forecasts):
            pred = model.predict(n=1, series=context_window)
            rolling_preds.append(pred)
            context_window = concatenate([context_window, pred])[-input_chunk_length:]

        rolling_pred_series = concatenate(rolling_preds)

        # Inverse transform predictions and actuals
        rolling_pred_series = scaler.inverse_transform(rolling_pred_series)
        val = scaler.inverse_transform(val)

        preds_df = rolling_pred_series.to_dataframe()
        preds_list = preds_df[col_name].tolist()
        all_combined_preds.append(preds_list)

        actuals_df = val.to_dataframe()
        actuals_list = actuals_df[col_name].tolist()
        all_combined_actuals.append(actuals_list)

        all_filenames.append(file)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Save predictions and actuals
with open('dlinear actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(all_filenames, all_combined_actuals):
        writer.writerow([file_name] + row)

with open('dlinear preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(all_filenames, all_combined_preds):
        writer.writerow([file_name] + row)