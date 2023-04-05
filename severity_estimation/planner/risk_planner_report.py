from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class RiskPlannerReport:
    risk_costs: Dict[int, Dict[str, np.ndarray]]
    compute_plan_runtimes: List[float]
    compute_trajectory_runtimes: List[float]
    prediction_runtimes: Dict[int, float]
    compute_cost_runtimes: Dict[int, Dict[str, float]]
    timesteps_us: Dict[int, float]

    def compute_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics over report fields.
        :return: dictionary containing summary statistics of each field.
        """
        return None
