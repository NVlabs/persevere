import numpy as np

from severity_estimation.hamilton_jacobi.hj_severity import HJSeverity

hj = HJSeverity()
hj.precompute()
hj.save("models/hj/hj_reachability.pkl")
