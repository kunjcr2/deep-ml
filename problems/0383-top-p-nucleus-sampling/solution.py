import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    probs = softmax(logits)
    val_idxs = sorted([(val, idx) for idx, val in enumerate(probs)], key=lambda x: x[0], reverse=True)

    selected = []
    cum_sum = 0.
    for val_idx in val_idxs:
        cum_sum += val_idx[0]
        selected.append(val_idx)

        if cum_sum >= p:
            break

    kept_probs = set([i[1] for i in selected])
    for i, prob in enumerate(probs):
        if i not in kept_probs:
            probs[i] = 0

    probs = [round(prob, 4) for prob in probs]
    probs = probs / sum(probs) # i did not know that normalizing means dividing by the sum

    return probs


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()