from sklearn.metrics.pairwise import cosine_similarity

def match(known, candidate, threshold=0.85):
    score = cosine_similarity(
        known.reshape(1, -1),
        candidate.reshape(1, -1)
    )[0][0]
    return score, score >= threshold
