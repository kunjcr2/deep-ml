import numpy as np
def train_neuron(
    features: np.ndarray, 
    labels: np.ndarray, 
    initial_weights: np.ndarray, 
    initial_bias: float, 
    learning_rate: float, 
    epochs: int
) -> (np.ndarray, float, list[float]):
    # Your code here

    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []
    N = len(features)

    for epoch in range(epochs):
        logits = features @ updated_weights + updated_bias

        logits = sigmoid(logits)
        
        errors = [logit-label for logit, label in zip(logits, labels)]
        loss = np.mean([er**2 for er in errors])

        mse_values.append(loss)

        # to get grads - dL/dw and dL/db
        delta = (2/N)*(np.array(errors))*(logits)*(1-logits)
        grad_w = features.T @ delta
        grad_b = sum(delta)

        updated_weights = updated_weights - learning_rate*grad_w
        updated_bias = updated_bias - learning_rate*grad_b

    updated_weights = [round(w, 4) for w in updated_weights]
    updated_bias = round(updated_bias, 4)
    mse_values = [round(m, 4) for m in mse_values]

    return updated_weights, updated_bias, mse_values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
