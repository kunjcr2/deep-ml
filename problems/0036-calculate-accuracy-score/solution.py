import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    n = len(y_true)
    t = 0
    for i in range(n):
        if y_true[i] == y_pred[i]:
            t += 1
        
    return t/n