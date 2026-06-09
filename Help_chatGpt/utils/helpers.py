# ============================================================
# utils/helpers.py
# ============================================================

import numpy as np

def confidence_interval(scores):

    mean = np.mean(scores)

    ci_low = np.percentile(scores, 2.5)

    ci_high = np.percentile(scores, 97.5)

    return mean, ci_low, ci_high
