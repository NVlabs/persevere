from collections import namedtuple

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

from severity_estimation.severity.bounds import lower_bound, upper_bound

RiskMeasure = namedtuple("RiskMeasure", ["LowerBound", "UpperBound"])


def risk(
    rx,
    ry,
    risk_aversion=0.95,
    confidence=0.9,
):
    n = len(rx)
    alpha = 1 - confidence
    Finv_x = lambda x: np.quantile(rx, q=x)
    Fy = ECDF(ry)
    high = upper_bound(risk_aversion, Finv_x, Fy, n, alpha)
    low = lower_bound(risk_aversion, Finv_x, Fy, n, alpha)
    return RiskMeasure(low, high)
