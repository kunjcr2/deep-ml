import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Your code here
    if Y is None:
        Y=X
    
    N = len(X)

    X_mean = np.mean(X, axis=0, keepdims=True)
    Y_mean = np.mean(Y, axis=0, keepdims=True)

    X_std = np.std(X, axis=0, keepdims=True)
    Y_std = np.std(Y, axis=0, keepdims=True)

    X_norm = (1/X_std)*(X-X_mean)
    Y_norm = (1/Y_std)*(Y-Y_mean)

    return (1/N)*(np.matmul(X_norm.T, Y_norm))
