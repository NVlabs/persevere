import argparse
import pickle
import sys
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Union

import msgpack
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from tqdm import tqdm

from severity_estimation.planner.risk_planner_results import RiskPlannerResults
from severity_estimation.severity.bound_risk_function import risk as bound_risk
from severity_estimation.severity.copula_risk_function import risk as copula_risk
from severity_estimation.severity.risk_threshold import RiskThreshold

fmt = (
    "<green>{time:MM.DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
)
logger.remove()  # All configured handlers are removed
logger.add(sys.stderr, format=fmt)


@dataclass(frozen=True)
class ExtRiskPlannerResults(RiskPlannerResults):
    has_collisions: Optional[int] = None
    earliest_collision_time: Optional[float] = None


class RiskType(Enum):
    BOUND = "bound"
    COPULA = "copula"


ConfusionMatrix = namedtuple("ConfusionMatrix", ["tp", "fp", "fn", "tn"])
SimpleRiskData = namedtuple("SimpleRiskData", ["risk", "confidence", "timing", "classification"])
BoundRiskData = namedtuple("BoundRiskData", ["lower", "upper", "timing", "classification"])
RiskData = Union[SimpleRiskData, BoundRiskData]
ParsedResults = Dict[str, Dict[str, ExtRiskPlannerResults]]


def classify(results, log, scenario, alg, param, group_by=1):
    vals = [elem.classification for elem in results.risk[log][scenario][alg][param].values()]
    # pred = any([sum([int(x) for x in g]) >= group_by for _, g in groupby(vals)])
    pred = False
    step = None
    for i in range(len(vals) - group_by + 1):
        if all(vals[i : i + group_by]):
            pred = True
            step = list(results.risk[log][scenario][alg][param].keys())[i + group_by - 1]
            break
    gt = bool(results.experiment_results[log][scenario].has_collisions)
    return pred, gt, step


def tree():
    # Shortcut to generate arbitrary nested defaultdicts
    return defaultdict(tree)


class PostprocessedResults:
    def __init__(self):
        self.experiment_results: ParsedResults
        self.risk_type: Optional[str] = None
        # [log][scenario][alg][param][step] -> RiskData
        self.risk: Dict[str, Dict[str, Dict[str, Dict[float, Dict[int, RiskData]]]]] = tree()
        # [alg][param] -> ConfusionMatrix
        self.confusion_matrix: Optional[Dict[str, Dict[float, ConfusionMatrix]]] = None

    def to_dict(self):
        return {
            "experiment_results": dict(self.experiment_results),
            "risk": dict(self.risk),
            "confusion_matrix": dict(self.confusion_matrix) if self.confusion_matrix else None,
        }


def parse_metrics(cfg, experiment_folder):
    collision_statistics = defaultdict(dict)
    metrics_directory = experiment_folder / cfg.results.metrics_dir
    for metrics_file in metrics_directory.iterdir():
        if metrics_file.is_file():
            with open(metrics_file, "rb") as input_file:
                metrics = pickle.load(input_file)
                for metric in metrics:
                    if metric["metric_computator"] == "collisions_statistics":
                        num_collision = metric["number_collisions_stat_value"]
                        earliest_collision_timestamp = metric["earliest_collision_timestamp_stat_value"]
                        assert len(num_collision) == 1, "There should be only one collision value"
                        # planner = metric["planner_name"]
                        log = metric["log_name"]
                        scenario = metric["scenario_name"]
                        collision_statistics[log][scenario] = (
                            num_collision[0],
                            earliest_collision_timestamp[0],
                        )
    return collision_statistics


def parse_results(cfg, experiment_folder) -> ParsedResults:
    collision_statistics = parse_metrics(cfg, experiment_folder)
    results_file = experiment_folder / cfg.results.result_file
    with open(results_file, "rb") as f:
        results = msgpack.load(f, strict_map_key=False)
    parsed = tree()  # defaultdict(lambda: defaultdict(dict))
    for log in results:
        for scenario in results[log]:
            data = results[log][scenario]
            planner_results = ExtRiskPlannerResults(
                start_time=data["start_time"],
                end_time=data["end_time"],
                compute_plan_runtimes=data["compute_plan_runtimes"],
                compute_trajectory_runtimes=data["compute_trajectory_runtimes"],
                prediction_runtimes=data["prediction_runtimes"],
                compute_cost_runtimes=data["compute_cost_runtimes"],
                risk_costs=data["risk_costs"],
                succeeded=data["succeeded"],
                timesteps_us=data["timesteps_us"],
                has_collisions=collision_statistics[log][scenario][0],
                earliest_collision_time=collision_statistics[log][scenario][1],
            )
            parsed[log][scenario] = planner_results
    return parsed


def _postprocess_copula_based_metric(
    results: PostprocessedResults,
    key: str,
    risk_aversion: List[float],
    risk_threshold: DictConfig,
):
    steps = [
        len(results.experiment_results[log][scenario].risk_costs)
        for log in results.experiment_results
        for scenario in results.experiment_results[log]
    ]
    n = len(risk_aversion) * sum(steps)
    th = risk_threshold
    with tqdm(total=n, desc=f"{key.upper(): <4}") as pbar:
        for q in risk_aversion:
            for log in results.experiment_results:
                for scenario in results.experiment_results[log]:
                    for step, data in results.experiment_results[log][scenario].risk_costs.items():
                        base = data[key]["base"]
                        hyp = data[key]["hyp"]
                        start_t = time.perf_counter()
                        risk = copula_risk(
                            base,
                            hyp,
                            risk_aversion=q,
                            risk_threshold=th,
                            copula="empirical",
                        )
                        end_t = time.perf_counter()
                        results.risk[log][scenario][key][q][step] = SimpleRiskData(
                            risk.risk, risk.confidence, end_t - start_t, risk.risk > th
                        )
                        pbar.update(1)


def _postprocess_bound_based_metric(
    results: PostprocessedResults,
    key: str,
    risk_aversion: List[float],
    confidence: float,
    risk_threshold: DictConfig,
):
    steps = [
        len(results.experiment_results[log][scenario].risk_costs)
        for log in results.experiment_results
        for scenario in results.experiment_results[log]
    ]
    n = len(risk_aversion) * sum(steps)
    th = RiskThreshold(risk_threshold.thresholds, risk_threshold.labels)
    with tqdm(total=n, desc=f"{key.upper(): <4}") as pbar:
        for q in risk_aversion:
            for log in results.experiment_results:
                for scenario in results.experiment_results[log]:
                    for step, data in results.experiment_results[log][scenario].risk_costs.items():
                        base = data[key]["base"]
                        hyp = data[key]["hyp"]
                        start_t = time.perf_counter()
                        risk = bound_risk(
                            base,
                            hyp,
                            risk_aversion=q,
                            confidence=confidence,
                        )
                        end_t = time.perf_counter()
                        results.risk[log][scenario][key][q][step] = BoundRiskData(
                            risk.LowerBound,
                            risk.UpperBound,
                            end_t - start_t,
                            risk.UpperBound >= th.thresholds[-1] and risk.LowerBound >= th.thresholds[-1],
                        )
                        pbar.update(1)


def _postprocess_hj(
    results: PostprocessedResults,
    key: str,
    risk_threshold: List[float],
):
    steps = [
        len(results.experiment_results[log][scenario].risk_costs)
        for log in results.experiment_results
        for scenario in results.experiment_results[log]
    ]
    n = len(risk_threshold) * sum(steps)
    key_base = "hj"
    # key_base = "hj_baseline"
    # key_hyp = "hj_hyp"
    with tqdm(total=n, desc=f"{key.upper(): <4}") as pbar:
        for t in risk_threshold:
            for log in results.experiment_results:
                for scenario in results.experiment_results[log]:
                    for step, data in results.experiment_results[log][scenario].risk_costs.items():
                        base = data[key]["base"]
                        hyp = data[key]["hyp"]
                        # Baseline
                        risk = min(base) if base else float("inf")
                        results.risk[log][scenario][key_base][t][step] = SimpleRiskData(risk, 1.0, 0, risk < t)
                        # # With Hypothesis
                        # risk = min(hyp) if hyp else float("inf")
                        # results.risk[log][scenario][key_hyp][t][step] = SimpleRiskData(
                        #     risk, 1.0, 0, risk < t
                        # )
                        pbar.update(1)


def _postprocess_collision_probability(
    results: PostprocessedResults,
    key: str,
    risk_aversion: List[float],
    risk_threshold: DictConfig,
):
    steps = [
        len(results.experiment_results[log][scenario].risk_costs)
        for log in results.experiment_results
        for scenario in results.experiment_results[log]
    ]
    n = len(risk_threshold.thresholds) * sum(steps)
    # th = RiskThreshold(risk_threshold.thresholds, risk_threshold.labels)
    with tqdm(total=n, desc=f"{key.upper(): <4}") as pbar:
        for q in risk_aversion:
            for log in results.experiment_results:
                for scenario in results.experiment_results[log]:
                    for step, data in results.experiment_results[log][scenario].risk_costs.items():
                        # fmt: off
                        base = max(data[key]["base"])
                        hyp = max(data[key]["hyp"])
                        results.risk[log][scenario][key][q][step] = BoundRiskData(base, hyp, 0.0, base > q and hyp > base)
                        # base_critical = th.membership(base) == th.highest
                        # hyp_critical = th.membership(hyp) == th.highest
                        # results.risk[log][scenario][f"{key}_xor"][q][step] = BoundRiskData(base, hyp, 0.0, not base_critical and hyp_critical)
                        # results.risk[log][scenario][f"{key}_hyp"][q][step] = BoundRiskData(base, hyp, 0.0, hyp_critical)
                        # results.risk[log][scenario][f"{key}_base"][q][step] = BoundRiskData(base, hyp, 0.0, base_critical)
                        # results.risk[log][scenario][f"{key}_or"][q][step] = BoundRiskData(base, hyp, 0.0, base_critical or hyp_critical)
                        # fmt: on
                        pbar.update(1)


def _compute_confusion_matrix(results: PostprocessedResults):
    n_scenarios = len([scenario for log in results.experiment_results for scenario in results.experiment_results[log]])
    if n_scenarios < 2:
        # Not enough scenarios to compute a confusion matrix
        logger.warning("Not enough scenarios to compute a confusion matrix")
        return
    results.confusion_matrix = tree()
    # [ALG][PARAM] -> {"pred": [], "gt": []}
    labels = defaultdict(lambda: defaultdict(lambda: {"pred": list(), "gt": list()}))
    for log in results.experiment_results:
        for scenario in results.risk[log]:
            for alg in results.risk[log][scenario]:
                for param in results.risk[log][scenario][alg]:
                    pred, gt, _ = classify(results, log, scenario, alg, param)
                    labels[alg][param]["pred"].append(pred)
                    labels[alg][param]["gt"].append(gt)
    for alg in labels:
        for param in labels[alg]:
            pred = labels[alg][param]["pred"]
            gt = labels[alg][param]["gt"]
            try:
                tn, fp, fn, tp = compute_confusion_matrix(gt, pred, labels=[False, True]).ravel()
            except ValueError:
                logger.warning("Something went wrong while computing the confusion matrix")
                results.confusion_matrix[alg][param] = None
            else:
                results.confusion_matrix[alg][param] = ConfusionMatrix(float(tp), float(fp), float(fn), float(tn))


def postprocess(cfg, experiment_folder: str):
    results = PostprocessedResults()
    results.experiment_results = parse_results(cfg, experiment_folder)
    if cfg.risk.copula.type == RiskType.COPULA.value:
        _postprocess_copula_based_metric(
            results=results,
            key="ttc",
            risk_aversion=cfg.risk.copula.risk_aversion,
            risk_threshold=cfg.risk.copula.threshold,
        )
        _postprocess_copula_based_metric(
            results=results,
            key="msd",
            risk_aversion=cfg.risk.copula.risk_aversion,
            risk_threshold=cfg.risk.copula.threshold,
        )
        results.risk_type = RiskType.COPULA
    elif cfg.risk.copula.type == RiskType.BOUND.value:
        _postprocess_bound_based_metric(
            results=results,
            key="ttc",
            risk_aversion=cfg.risk.copula.risk_aversion,
            confidence=cfg.risk.copula.confidence,
            risk_threshold=cfg.risk.copula.threshold,
        )
        _postprocess_bound_based_metric(
            results=results,
            key="msd",
            risk_aversion=cfg.risk.copula.risk_aversion,
            confidence=cfg.risk.copula.confidence,
            risk_threshold=cfg.risk.copula.threshold,
        )
        results.risk_type = RiskType.BOUND
    else:
        raise ValueError(f"Unknown risk type: {cfg.risk.copula.type}")
    _postprocess_hj(results, "hj", cfg.risk.hj.thresholds)
    _postprocess_collision_probability(results, "cp", cfg.risk.cp.risk_aversion, cfg.risk.cp.threshold)
    _compute_confusion_matrix(results)
    return results


def main_app() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()

    experiment_folder = args.experiment
    if not experiment_folder.is_dir():
        experiment_folder = "experiments" / experiment_folder
    if not experiment_folder.is_dir():
        print(f"Could not find experiment folder: {experiment_folder}")
        exit(1)
    print("Using experiment folder:", experiment_folder)

    cfg_path = experiment_folder / "config.yaml"
    logger.debug("Using config:", cfg_path)
    cfg = OmegaConf.load(cfg_path)
    logger.info(f"Starting postprocessing {experiment_folder}")

    postprocessed_results = postprocess(cfg, experiment_folder)

    with open(experiment_folder / cfg.results.postprocessed_file, "wb") as handle:
        pickle.dump(postprocessed_results, handle)

    logger.info(f"Done!\nYou can visualize the results by running:\n > poetry run viz {experiment_folder} --nuboard")


if __name__ == "__main__":
    main_app()
