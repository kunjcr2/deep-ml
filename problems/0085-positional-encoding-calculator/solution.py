import numpy as np

def pos_encoding(position: int, d_model: int):
    # Your code here
    if position == 0 or d_model <= 0:
        return -1

    # generating a buffer kind of thing with dim (pos, 1)
    pos_encodings_buff = np.arange(0, position)
    pos_encodings_buff = np.expand_dims(pos_encodings_buff, axis=-1) # (pos, 1)
    
    # these are angle factors from rope with dim (1, d_model // 2)
    factors = np.exp(np.arange(0, d_model, 2) * -(np.log(np.array(10000.0)) / d_model)) 
    factors = np.transpose(np.expand_dims(factors, axis=-1))

    # generating angles at (pos, d_model // 2)
    angles = pos_encodings_buff @ factors

    # angles for each
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    # adding in the sin and cos and interleaving them for positional encodings
    pos_encodings = np.zeros((position, d_model))
    pos_encodings[:, 0::2] = sin_vals
    pos_encodings[:, 1::2] = cos_vals

    pos_encodings = np.float16(pos_encodings)
    return pos_encodings
