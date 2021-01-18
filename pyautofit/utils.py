import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score

__all__ = ['woe', 'iv', 'gini']

def woe(g_pct, b_pct):
    result = np.log(g_pct/b_pct)
    return np.where(result == np.inf, 0, result)


def iv(g_pct, b_pct):
    return np.sum((g_pct - b_pct) * woe(g_pct, b_pct))


def gini(y_true, y_pred, sample_weight=None):
    """Calcuate GINI score for a result
    """
    roc_auc = roc_auc_score(y_true, y_pred, sample_weight)
    result = 2 * roc_auc - 1
    return result
