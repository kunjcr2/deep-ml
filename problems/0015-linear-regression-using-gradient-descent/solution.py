import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    # Your code here: implement gradient descent

    for _ in range(iterations):
        loss = get_loss(X, y, theta)
        theta = get_update(X, y, loss, alpha, theta)

    return [round(t, 4) for t in theta.flatten()]

def get_loss(X, y, theta):
    X_T = np.transpose(X)
    pred = np.matmul(X, theta)
    loss = (1/2*len(X))*(sum((pred - y)**2))

    return loss

def get_update(X, y, loss, alpha, theta):

    grad = (1/len(X))*(np.transpose(X) @ (np.matmul(X, theta) - y))
    theta = theta - alpha*grad

    return theta
