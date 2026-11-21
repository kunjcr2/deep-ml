import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

    if np.linalg.det(np.array(T)) == 0 or np.linalg.det(np.array(S)) == 0:
        return -1

    transformed_matrix = np.matmul(np.linalg.inv(T), A)
    transformed_matrix = np.matmul(transformed_matrix, S)

    return transformed_matrix