import numpy as np


def predict(**kwargs):
    return {
        "class": int(np.random.randint(0, 2)),
        "embedding": np.random.random(size=(1, 100)),
        "confidence": 1,
    }
