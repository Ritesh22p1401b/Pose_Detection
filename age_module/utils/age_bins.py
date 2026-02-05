AGE_BINS = [
    (0, 2), (3, 5), (6, 8), (9, 12),
    (13, 17), (18, 24), (25, 32),
    (33, 40), (41, 50), (51, 60),
    (61, 70), (71, 100)
]

BIN_CENTERS = [(low + high) / 2 for low, high in AGE_BINS]

def age_to_class(age: int):
    for i, (low, high) in enumerate(AGE_BINS):
        if low <= age <= high:
            return i
    return None
