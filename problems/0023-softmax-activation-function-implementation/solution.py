import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)

    e_scores = [math.exp(score - max_score) for score in scores]
    e_scores_sum = sum(e_scores)

    return [e_score/(e_scores_sum + 1e-9) for e_score in e_scores]