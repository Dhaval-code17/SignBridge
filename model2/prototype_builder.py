"""
Model 2: Sentence Prototype Matcher & Vector Similarity Search.
Loads 512-D sentence prototypes and computes cosine similarity against input query embeddings.
"""

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Computes L2 normalization of input vector/matrix."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def match_prototypes(query_vector: np.ndarray, reference_matrix: np.ndarray, reference_glosses: list, top_k: int = 5):
    """
    Computes cosine similarity of query 512-D embedding against reference matrix prototypes.
    Returns sorted top_k predictions.
    """
    query = l2_normalize(query_vector)
    scores = reference_matrix @ query

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [
        {
            "gloss": reference_glosses[i],
            "similarity": round(float(scores[i]), 4),
        }
        for i in top_indices
    ]
    return results
