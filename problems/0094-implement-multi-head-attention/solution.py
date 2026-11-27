import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return X@W_q, X@W_k, X@W_v

def softmax(x):
    shifted = x - np.max(x)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    attn_scores = (Q @ K.T) / np.sqrt(K.shape[-1])
    attn_weights = [softmax(attn_score) for attn_score in attn_scores]

    out = attn_weights @ V

    return out

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    S, D = Q.shape

    Q = np.reshape(Q, (S, n_heads, -1)).transpose(1,0,2)
    K = np.reshape(K, (S, n_heads, -1)).transpose(1,0,2)
    V = np.reshape(V, (S, n_heads, -1)).transpose(1,0,2)

    attn_scores = Q @ K.transpose(0, 2, 1) / np.sqrt(K.shape[-1])
    attn_weights = []
    for n in range(n_heads):
        temp = []
        for i in attn_scores[n]:
            temp.append(softmax(i))
        attn_weights.append(temp)
    
    attn_weights = np.array(attn_weights)

    out = (attn_weights @ V).transpose(1, 0, 2).reshape((S, D))

    return out
