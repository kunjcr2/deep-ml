import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    # Your code here
    ans1 = []
    ans2 = []
    for i in range(0, len(X), batch_size):
        # i keep messing this part up in terms of making np.float64 to int
        ans1.append([[int(v) for v in x] for x in X[i:i+batch_size]])
        if y is not None:
            ans2.append(list(y[i:i+batch_size]))

    ans = []
    if len(ans2) != 0:
        for o, t in zip(ans1, ans2):
            ans.append([o, t])
    else:
        ans = ans1

    return ans