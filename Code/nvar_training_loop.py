import os
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
current_dir = os.path.dirname(os.path.abspath(__file__))
merged_folder_path = os.path.join(current_dir, '..', 'Epicasting')
merged_folder_path = os.path.join(merged_folder_path, 'Merged')
merged_folder_path = os.path.abspath(merged_folder_path)

extensions = ('*.xlsx', '*.xls')

excel_files = []
for ext in extensions:
    excel_files.extend(glob.glob(os.path.join(merged_folder_path, ext)))

num_lags = 10
poly_degree = 2
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
    try:
        data_df = pd.read_excel(file)
        
        for col in cols:
            if col in data_df.columns:
                col_name = col
        data = data_df[col_name].values.reshape(-1, 1)  # shape: (T, 1)

        X_hist = create_history_matrix(data, num_lags)
        y_target = create_target_vector(data, num_lags)

        X_train, X_test, y_train, y_test = train_test_split(X_hist, y_target, test_size=0.1, shuffle=False)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        # Fit Ridge regression model
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_poly, y_train)

        y_test_pred = model.predict(X_test_poly)
        nvar_all_smape.append(smape(y_test, y_test_pred))
        file_names_evaluated.append(file)
    except Exception as e:
        print(f"Error occured in file: {file} is = {e}")

df = pd.DataFrame({'File name': file_names_evaluated, 'sMAPE': nvar_all_smape})
df.to_csv('NVAR sMAPE.csv', index=False)