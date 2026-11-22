import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here

    standardized_data = get_std(data)
    normalized_data = get_minmax(data)

    return standardized_data, normalized_data

def get_std(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    return (x-mean)/(std)

def get_minmax(x):
    mx = np.max(x, axis=0, keepdims=True)
    mn = np.min(x, axis=0, keepdims=True)

    return (x-mn)/(mx-mn)