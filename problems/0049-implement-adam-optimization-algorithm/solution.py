import torch

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10) -> torch.Tensor:
    """
    Implements Adam optimization algorithm using PyTorch's built-in optimizer.

    Args:
        f: The objective function to be optimized
        grad: A function that computes the gradient (unused; autograd is used instead)
        x0: Initial parameter values (torch.Tensor)
        learning_rate: The step size (default: 0.001)
        beta1: Exponential decay rate for the first moment estimates (default: 0.9)
        beta2: Exponential decay rate for the second moment estimates (default: 0.999)
        epsilon: A small constant for numerical stability (default: 1e-8)
        num_iterations: Number of iterations to run the optimizer (default: 10)

    Returns:
        torch.Tensor: Optimized parameters
    """
    # Your code here
    mom = 0
    vel = 0

    for t in range(num_iterations):

      g = grad(x0)

      mom = beta1*mom + (1-beta1)*g
      vel = beta2*vel + (1-beta2)*(g**2)

      # GPT said that beta needs to be powered by iteration number <- Did not know that
      mom_hat = mom/(1-beta1**(t+1))
      vel_hat = vel/(1-beta2**(t+1))

      x0 = x0 - learning_rate*(mom_hat/((vel_hat**0.5)+epsilon))

    return x0