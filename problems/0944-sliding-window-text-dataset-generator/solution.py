def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    if len(token_ids)<stride:
        return []
        
    data = []
    labels = []
    for i in range(0, len(token_ids), stride):
        data.append(token_ids[i:i+max_length])
        labels.append(token_ids[i+1:i+max_length+1])

    ans = []
    n = len(data[0])
    for d, l in zip(data, labels):
        if len(d)<n or len(l)<n:
            break
        ans.append((d, l))

    return ans