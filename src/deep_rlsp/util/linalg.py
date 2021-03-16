import numpy as np


# @memoize  # don't memoize for continuous vectors, leads to memory leak
def get_cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
