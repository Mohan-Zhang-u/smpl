import numpy as np


# fit calculation
def fitting_score(
        y,
        yhat
):
    ey = y - yhat
    em = y - np.mean(y)
    return 100.0 * (1 - np.linalg.norm(ey) / np.linalg.norm(em))


# mean squared error
def get_mse(
        y,
        yhat
):
    ey = y - yhat
    N = len(y)
    return (1 / N) * np.matmul(ey, ey)


# Kalman filter
def state_estimator(
        x,
        u,
        ym,
        A,
        B,
        C,
        K
):
    xe = np.matmul(A, x) \
         + np.matmul(B, u) \
         + np.matmul(K, ym - np.matmul(C, x))
    return np.array(xe).ravel()
