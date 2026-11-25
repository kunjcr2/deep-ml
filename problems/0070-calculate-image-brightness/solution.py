import numpy as np

def calculate_brightness(img):
    # Write your code here
    if len(img) == 0:
        return -1

    total_len = set([len(i) for i in img])
    if len(total_len) > 1:
        return -1

    for r in img:
        for c in r:
            if c < 0 or c > 255:
                return -1
    
    row_min = np.mean(img, axis=-1)
    return np.mean(row_min)