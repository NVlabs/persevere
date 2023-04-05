from collections import namedtuple

import numpy as np
import pandas as pd
from copulae import (
    ClaytonCopula,
    EmpiricalCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentCopula,
    pseudo_obs,
)
from scipy import optimize
from statsmodels.distributions.empirical_distribution import ECDF

from severity_estimation.severity.bounds import lower_bound, upper_bound

RiskMeasure = namedtuple("RiskMeasure", ["risk", "confidence"])


def _find_confidence(r, q, Finv_x, Fy, n, th=0.7):
    zero_prob = 1e-8
    sure_prob = 1 - zero_prob
    if r >= th:
        # max confidence s.t. Low bound >= th
        f = lambda x: lower_bound(q, Finv_x, Fy, n, 1 - x) - th
        if f(sure_prob) > 0:
            return sure_prob
        if f(zero_prob) <= 0:
            return zero_prob
    else:
        # max confidence s.t. up bound < th
        f = lambda x: upper_bound(q, Finv_x, Fy, n, 1 - x) - th
        if f(sure_prob) < 0:
            return sure_prob
        if f(zero_prob) >= 0:
            return zero_prob
    return optimize.bisect(f, zero_prob, sure_prob)


def risk(rx, ry, risk_aversion=0.95, risk_threshold=0.7, copula="empirical"):
    n = len(rx)
    df = pd.DataFrame.from_dict({"x": rx, "y": ry})
    u = pseudo_obs(df)  # data must be converted to pseudo observations
    Finv_x = lambda x: np.quantile(rx, q=x)
    Fy = ECDF(ry)
    if copula == "empirical" or copula == "beta":
        cc = EmpiricalCopula(u, smoothing="beta")
    elif copula == "gaussian":
        cc = GaussianCopula(dim=2)  # initializing the copula
        cc.fit(u, method="ml", verbose=0)  # fit the copula to the data
    elif copula == "gumbel":
        cc = GumbelCopula(dim=2)
        cc.fit(u, method="ml", verbose=0)  # { 'ml', 'irho', 'itau' }
    else:
        raise ValueError("Unknown copula type")

    ux = risk_aversion
    uy = Fy(Finv_x(ux))
    r = np.clip(1 - cc.cdf((ux, uy)) / ux, 0.0, 1.0)
    confidence = _find_confidence(r, risk_aversion, Finv_x, Fy, n, risk_threshold)
    return RiskMeasure(r, confidence)
