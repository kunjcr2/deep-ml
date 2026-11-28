import numpy as np
def precision(y_true, y_pred):
    # Your code here
    # precision is how many that were flagged are true!

    total_hits = sum(y_pred)
    total_matches = 0
    for t, p in zip(y_true, y_pred):
        if t==1 and p==1:
            total_matches += 1

    return total_matches/total_hits if total_hits!=0 else 0.0