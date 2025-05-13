import numpy as np

def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    denominator[denominator == 0] = 1e-8  # avoid divide-by-zero
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denominator)