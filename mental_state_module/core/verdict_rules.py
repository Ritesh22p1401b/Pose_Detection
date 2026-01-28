from mental_state_module.config import (
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
    HIGH_RISK_THRESHOLD,
    DISTRESS_THRESHOLD,
)


def classify(first_emotion, emotions):
    neg = sum(1 for e in emotions if e in NEGATIVE_EMOTIONS)
    pos = sum(1 for e in emotions if e in POSITIVE_EMOTIONS)

    if neg >= HIGH_RISK_THRESHOLD:
        return "Mentally Unstable (High Risk)"

    if first_emotion in NEGATIVE_EMOTIONS and neg >= DISTRESS_THRESHOLD:
        return "Emotionally Distressed"

    if pos >= DISTRESS_THRESHOLD:
        return "Mentally Stable"

    return "Needs Further Observation"
