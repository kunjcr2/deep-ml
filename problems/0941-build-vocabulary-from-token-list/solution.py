def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    tokens = sorted(set(tokens))
    d = {}

    for i, token in enumerate(tokens):
        d[token] = i

    return d