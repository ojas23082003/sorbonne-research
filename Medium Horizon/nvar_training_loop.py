import os
import csv
import glob
import torch
import numpy as np
import pandas as pd
from nbeats import *
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from smape import smape
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


nvar_all_smape = []
file_names_evaluated = []
all_predictions_combined = []
all_ground_truths_combined = []
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

num_lags = 10
poly_degree = 1
ridge_alpha = 1e-4

def create_history_matrix(data, num_lags):
    N, d = data.shape
    X = []
    for t in range(num_lags, N):
        h = data[t-num_lags:t].flatten()
        X.append(h)
    return np.array(X)

def create_target_vector(data, num_lags):
    return data[num_lags:]

cols = ["Cases","Total_cases","cases"]

for file in excel_files:
    # try:
    data_df = pd.read_excel(file)
    
    for col in cols:
        if col in data_df.columns:
            col_name = col
    data = data_df[col_name].values
    data = data.tolist()

    train_data = np.array(data[:int(0.95*len(data))]).reshape(-1, 1)  # shape: (T, 1)
    test_data = data[int(0.95*len(data)):]

    X_hist = create_history_matrix(train_data, num_lags)
    y_target = create_target_vector(train_data, num_lags)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_hist)
    # X_test_scaled = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train_scaled)
    # X_test_poly = poly.transform(X_test_scaled)

    # Fit Ridge regression model
    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train_poly, y_target)

    #### prediction loop ####
    last_known_input = X_hist[-1].flatten()  # shape: (num_lags,)
    last_known_target = y_target[-1][0]           # true next value

    initial_window = last_known_input[1:]
    initial_window = np.append(initial_window,last_known_target)
    last_window = initial_window.reshape(1,-1)
    rolling_predictions = []
    for _ in range(len(test_data)):
        
        last_window_scaled = scaler.transform(last_window)
        # Apply polynomial transformation
        last_window_poly = poly.transform(last_window_scaled)   

        # Predict the next value
        print(last_window_poly)
        next_pred_scaled = model.predict(last_window_poly)[0]

        # Save the prediction (inverse transform)
        dummy_input = np.zeros((1, 10))
        dummy_input[0, -1] = next_pred_scaled
        next_pred = scaler.inverse_transform(dummy_input)
        next_pred = next_pred[0,-1]
        rolling_predictions.append(next_pred)

        # Scale the prediction and update the rolling window
        # next_pred_input_scaled = scaler.transform([[next_pred]])[0][0]
        new_window_flat = np.append(last_window.flatten()[1:], next_pred)
        last_window = new_window_flat.reshape(1, -1)
    #### pred loop ends ####
    nvar_all_smape.append(smape(test_data, rolling_predictions))
    all_predictions_combined.append(rolling_predictions)
    all_ground_truths_combined.append(test_data)
    file_names_evaluated.append(file)
    # except Exception as e:
    #     print(f"Error occured in file: {file} is = {e}")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': nvar_all_smape})
df.to_csv('NVAR sMAPE.csv', index=False)

with open('nvar actuals.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_ground_truths_combined):
        writer.writerow([file_name] + row)

with open('nvar preds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for file_name, row in zip(file_names_evaluated, all_predictions_combined):
        writer.writerow([file_name] + row)