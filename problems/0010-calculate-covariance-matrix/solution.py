import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    # Your code here
    # apparently you normalize and then do dot between all <- thats how covariance matrices are built faster

    means = np.mean(vectors, axis=-1, keepdims=True)
    norm_vectros = vectors - means
    M = len(vectors)
    N = len(vectors[0])
    cov_mat = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            cov_mat[i][j] = (1/(N-1))*(np.dot(norm_vectros[i], norm_vectros[j]))

    # print(cov_mat)
    return cov_mat