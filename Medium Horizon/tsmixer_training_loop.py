import os
import csv
import glob
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from darts.models import TSMixerModel
import torch.nn as nn
from darts import concatenate
# from darts.utils.forecasting import ForecastingModel
# from darts.utils.utils import series2array

# Load CSV (without datetime column)
# df = pd.read_excel("Ahmedabad_Dengue.xlsx")  # change filename

# Option 1: Use integer index as synthetic datetime (e.g., daily)

all_combined_preds = []
all_combined_actuals = []
all_filenames = []

current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')
cols = ["Cases","Total_cases","cases"]

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))
# Optional: Scale the series
for file in excel_files:
    # try:
    df = pd.read_excel(file)

    # Step 2: Extract the time series column
    for col in cols:
        if col in df.columns:
            col_name = col
            break

    series = TimeSeries.from_dataframe(df, value_cols=col_name, freq="D")
    scaler = Scaler(StandardScaler())
    series_scaled = scaler.fit_transform(series)

    train, val = series_scaled.split_before(0.95)

    model = TSMixerModel(
        input_chunk_length=3,
        output_chunk_length=1,
        hidden_size=64,
        dropout=0.1,
        batch_size=16,
        n_epochs=30,
        optimizer_kwargs={"lr": 1e-3},
        loss_fn=nn.MSELoss(),
        pl_trainer_kwargs={"accelerator": "auto"},
        random_state=42,
    )

    model.fit(train, verbose=True)

    print(f"going for rolling window")
    context_window = train[-3:]  

    # Container for predictions
    rolling_preds = []

    # Number of predictions = length of validation set
    n_forecasts = len(val)

    for i in range(n_forecasts):
        # Predict next step using current context window
        pred = model.predict(n=1, series=context_window)

        # Save prediction
        rolling_preds.append(pred)

        # Append prediction to context window and remove oldest value
        context_window = concatenate([context_window, pred])[-3:]

    # Combine predictions into a TimeSeries
    # from darts.utils.utils import combine_series
    rolling_pred_series = concatenate(rolling_preds)

    # Inverse transform predictions and validation series
    rolling_pred_series = scaler.inverse_transform(rolling_pred_series)
    val_actual_series = scaler.inverse_transform(val)

    
    preds_df = rolling_pred_series.to_dataframe()
    preds_list = preds_df[col_name].tolist()
    all_combined_preds.append(preds_list)
    
    actuals_df = val_actual_series.to_dataframe()
    actuals_list = actuals_df[col_name].tolist()
    all_combined_actuals.append(actuals_list)
    all_filenames.append(file)
    # except Exception as e:
    #     print(f"Error processing {file}: {e}")


with open('tsmixer actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(all_filenames, all_combined_actuals):
        writer.writerow([file_name] + row)

with open('tsmixer preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(all_filenames, all_combined_preds):
        writer.writerow([file_name] + row)