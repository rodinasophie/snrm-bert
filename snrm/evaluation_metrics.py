import numpy as np


def retrieval_score(q, d):
    return np.dot(q, d)
