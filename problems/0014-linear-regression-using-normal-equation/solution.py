import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:

    X_T = np.transpose(X)
    t_1 = np.matmul(X_T, X)

    t_2 = np.matmul(X_T, y)
    t_1_inv = np.linalg.inv(t_1)

    theta = np.matmul(t_1_inv, t_2)

    return theta