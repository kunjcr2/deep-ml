import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Your code here
    sp = np.log(1 + np.exp(X @ W_noise))
    H = X @ W_g + N * sp
    
    top_k = get_topk(H, k)
    
    ans = np.zeros_like(H)
    for i in range(len(top_k)):
        for u in top_k[i]:
            ans[i][u[1]] = u[0]

    ans = [list(softmax(a)) for a in ans]

    return ans

def softmax(x):
    shifted = x - np.max(x)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)

def get_topk(x, k):
    temp = []
    for h in x:
        temp2 = sorted([(u,i) for i,u in enumerate(h)], reverse=True)
        temp3 = []

        for i in range(len(h)):
            if i<k:
                temp3.append(temp2[i])
            else:
                temp4 = list(temp2[i])
                temp4[0] = float('-inf')
                temp3.append(tuple(temp4))
        temp.append(temp3)
    
    return temp
