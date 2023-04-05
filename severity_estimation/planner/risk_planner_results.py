from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class RiskPlannerResults:
    start_time: float
    end_time: float
    compute_plan_runtimes: List[float]
    compute_trajectory_runtimes: List[float]
    prediction_runtimes: Dict[int, float]
    compute_cost_runtimes: Dict[int, Dict[str, float]]
    risk_costs: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    succeeded: bool
    timesteps_us: Dict[int, float]
