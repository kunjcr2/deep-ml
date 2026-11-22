import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # GPT = GPT Helped finding the bug = 2 in this code

    x = X
    t = x
    
    # earlier mistake was: I was doing, 
    # x = x + LayerNorm(Attn(x))
    # instead of
    # x = LayerNorm(x + Attn(x))
    # GPT^^^
    
    x = t + attention(x, weights, num_heads)
    x = norm(x, weights, eps, "f")

    t = x
    x = t + ffn(x, weights)
    x = norm(x, weights, eps, "s")

    return x

def attention(x, weights, num_heads):
    b, s, d = x.shape

    # query, key, values
    Q = np.matmul(x, weights["W_q"])
    K = np.matmul(x, weights["W_k"])
    V = np.matmul(x, weights["W_v"])

    # num_heads breaking
    Q = Q.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)
    K = K.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)
    V = V.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)

    # attn scores
    attn_scores = Q @ K.transpose(0, 1, 3, 2)
    d_k = K.shape[-1]**0.5
    attn_weights = softmax(attn_scores / d_k)

    # logits
    hidden = attn_weights @ V
    hidden = hidden.transpose(0, 2, 1, 3)
    hidden = np.reshape(hidden, (b, s, d))
    out = hidden @ weights["W_o"]

    return out

def norm(x, weights, eps, f_s):
    gamma = weights["gamma1"] if f_s=="f" else weights["gamma2"]
    beta = weights["beta1"] if f_s=="f" else weights["beta2"]

    # keepdims=True to keep the (B, S, 1) instead of just (B, S) <-- GPT
    m = np.mean(x, axis=-1, keepdims=True) 
    s = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - m)/np.sqrt(s + eps)

    return gamma * x_norm + beta

def ffn(x, weights):
    hidden = np.matmul(x, weights["W1"]) + weights["b1"]
    hidden = np.maximum(0, hidden)
    out = np.matmul(hidden, weights["W2"]) + weights["b2"]

    return out

def softmax(x):
    """external tool used"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)