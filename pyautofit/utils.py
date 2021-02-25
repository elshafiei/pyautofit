import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score

import subprocess
import io
import json
import pandas as pd

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


def query_impala_table(qry, impala_server = 'asprlxhd401', schema='autelco_poc_db', sep='\t'):
    cmd = ['impala-shell', '-k', '-B', '--print_header', '-i', impala_server, '-q', qry, '-d', schema]
    try:
        ret = subprocess.check_output(cmd, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(e.output)
        return False

    df = pd.read_csv(io.StringIO(ret), sep=sep)
    return df
