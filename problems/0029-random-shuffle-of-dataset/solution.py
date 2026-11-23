import numpy as np

def shuffle_data(X, y, seed=None):
    # Your code here
    if seed is not None:
        np.random.seed(seed)

    combined = [(X_u, y_u) for X_u, y_u in zip(X, y)]
    np.random.shuffle(combined)
    
    X, y = [], []
    for X_u, y_u in combined:
        X.append(X_u)
        y.append(y_u)

    return X, y