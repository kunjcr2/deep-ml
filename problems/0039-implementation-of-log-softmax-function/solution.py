import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    max_score = max(scores)
    e_powers = [np.exp(score-max_score) for score in scores]
    e_power_sum = sum(e_powers)

    return [np.log(e_power/e_power_sum) for e_power in e_powers]