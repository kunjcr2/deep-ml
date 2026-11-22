import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Your code here
    mse = 0
    activations = np.matmul(features, weights)
    activations = activations + bias
    probabilities = [round(sigmoid(activation), 4) for activation in activations]

    errors = [abs(p-t) for p, t in zip(probabilities, labels)]
    mse = round(np.mean([error**2 for error in errors]), 4)

    # squared_probs = [prob**2 for prob in probabilities]
    # root_mean_squared_probs = np.mean(squared_probs)

    return probabilities, mse

def sigmoid(x):
    return 1 / (1 + math.exp(-x))