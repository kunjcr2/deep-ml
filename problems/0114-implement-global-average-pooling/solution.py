import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    # Your code here
    x = np.mean(np.mean(x.transpose(2,0,1), axis=-1), axis=-1)
    return x