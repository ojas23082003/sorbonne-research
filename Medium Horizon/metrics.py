import numpy as np
from properscoring import crps_ensemble

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def mase(y_true, y_pred, seasonality=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = y_true.shape[0]

    # Naive forecast: shift by seasonality
    naive_forecast = y_true[:-seasonality]
    naive_diff = np.abs(y_true[seasonality:] - naive_forecast)
    scale = np.mean(naive_diff)

    errors = np.abs(y_true - y_pred)
    return np.mean(errors) / scale if scale != 0 else np.inf

def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def crps(y_true, ensemble_forecasts):
    """
    y_true: array of shape (n_samples,)
    ensemble_forecasts: array of shape (n_samples, n_ensemble_members)
    """
    y_true = np.array(y_true)
    ensemble_forecasts = np.array(ensemble_forecasts)
    return np.mean(crps_ensemble(y_true, ensemble_forecasts))