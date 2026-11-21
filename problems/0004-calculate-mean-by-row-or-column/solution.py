import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    dim = int(mode != "column")
    means = np.mean(matrix, axis=dim)
    return means