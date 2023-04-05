import argparse
import pickle
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from postprocessor import PostprocessedResults, SimpleRiskData, classify

mpl.style.use("seaborn")
sns.set()
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from omegaconf import OmegaConf
from rich import print as rprint
from rich.console import Console
from rich.table import Table


def _std_stats(values):
    return (
        {
            "mean": np.mean(values),
            "std": np.std(values),
            "median": np.median(values),
        }
        if values
        else {"mean": np.nan, "std": np.nan, "median": np.nan}
    )


def _timing_info(cfg, results):
    timing = defaultdict(
        lambda: defaultdict(lambda: {"total": [], "risk": [], "prediction": []})
    )
    for log in results.risk:
        for scenario in results.risk[log]:
            cost_runtimes = results.experiment_results[log][
                scenario
            ].compute_cost_runtimes
            pred_runtime = results.experiment_results[log][scenario].prediction_runtimes
            for alg in results.risk[log][scenario]:
                for param in results.risk[log][scenario][alg]:
                    risk_values = results.risk[log][scenario][alg][param]
                    timing[alg][param]["risk"].extend(
                        [
                            cost_runtimes[step][alg] + risk_values[step].timing
                            for step in risk_values.keys()
                        ]
                    )
                    if alg == "hj":
                        timing[alg][param]["total"].extend(
                            [
                                risk_values[step].timing + cost_runtimes[step][alg]
                                for step in risk_values.keys()
                            ]
                        )
                    else:
                        timing[alg][param]["prediction"].extend(
                            [pred_runtime[step] for step in risk_values.keys()]
                        )
                        timing[alg][param]["total"].extend(
                            [
                                risk_values[step].timing
                                + cost_runtimes[step][alg]
                                + pred_runtime[step]
                                for step in risk_values.keys()
                            ]
                        )
    return timing


def _detection_info(crg, results):
    # Alarm to Collision
    alarm_to_collision = defaultdict(lambda: defaultdict(list))
    for log in results.experiment_results:
        for scenario in results.risk[log]:
            for alg in results.risk[log][scenario]:
                for param in results.risk[log][scenario][alg]:
                    pred, gt, step = classify(results, log, scenario, alg, param)
                    if pred and gt:
                        # fmt: off
                        earliest_collision = results.experiment_results[log][scenario].earliest_collision_time
                        earliest_alarm =results.experiment_results[log][scenario].timesteps_us[step]
                        alarm_to_collision[alg][param].append((earliest_collision - earliest_alarm)/1e6) 
                        # fmt: on
    # Stats
    stats = defaultdict(lambda: defaultdict(dict))
    for log in results.experiment_results:
        for scenario in results.experiment_results[log]:
            for alg in results.risk[log][scenario]:
                for param in results.risk[log][scenario][alg]:
                    if results.confusion_matrix is not None:
                        cm = results.confusion_matrix[alg][param]
                        try:
                            precision = cm.tp / (cm.tp + cm.fp)
                        except ZeroDivisionError:
                            precision = np.nan
                        try:
                            recall = cm.tp / (cm.tp + cm.fn)
                        except:
                            recall = np.nan
                        try:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                        except ZeroDivisionError:
                            f1_score = np.nan
                        try:
                            accuracy = (cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn)
                        except ZeroDivisionError:
                            accuracy = np.nan
                    else:
                        precision = recall = f1_score = accuracy = np.nan
                    stats[alg][param] = {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "accuracy": accuracy,
                        "alarm-to-collision": alarm_to_collision[alg][param],
                    }
    return stats


def _compute_summary(cfg, results):
    timing_info = _timing_info(cfg, results)
    detection = _detection_info(cfg, results)
    summary = defaultdict(lambda: defaultdict(dict))
    for alg in detection:
        for param in detection[alg]:
            summary[alg][param] = dict(detection[alg][param])
            summary[alg][param]["timing"] = {
                k: _std_stats(v) for k, v in timing_info[alg][param].items()
            }
            summary[alg][param]["alarm-to-collision"] = _std_stats(
                summary[alg][param]["alarm-to-collision"]
            )
    return summary


def generate_summary(cfg, results, experiment_folder):
    def _prt(val, best=None, fmt=".2f"):
        if best is not None and val == best:
            return f"[bold]{val:{fmt}}[/bold]"
        else:
            return f"{val:{fmt}}"

    console = Console()
    rprint(f"[bold magenta]Num. Scenarios[/bold magenta]:  {len(cfg.scenarios)}")
    # Hypothesis generator
    if cfg.hypothesis_generator.use_ground_truth:
        rprint("[bold magenta]Hyp. Generator[/bold magenta]:  Ground Truth")
    else:
        keys = [
            str(k).replace("_", " ").capitalize()
            for k in cfg.hypothesis_generator.keys()
            if k != "use_ground_truth"
        ]
        rprint(f"[bold magenta]Hyp. Generator[/bold magenta]:  {keys}")
    # Risk Configuration
    rprint(
        f"[bold magenta]Risk Type[/bold magenta]: {' '*6}{cfg.risk.copula.type.capitalize()}"
    )
    stats = _compute_summary(cfg, results)
    # Runtime Table
    table = Table(title="Runtime")
    table.add_column("Algorithm", justify="left", no_wrap=True)
    table.add_column("Param", justify="center")
    table.add_column("Total", justify="center")
    table.add_column("Prediction", justify="center")
    table.add_column("Cost", justify="center")
    table.add_column("Risk", justify="center")
    for alg in sorted(stats.keys(), reverse=True):
        for param in sorted(stats[alg].keys(), reverse=True):
            tot = stats[alg][param]["timing"]["total"]["mean"]
            pred = stats[alg][param]["timing"]["prediction"]["mean"]
            cost = stats[alg][param]["timing"]["cost"]["mean"]
            risk = stats[alg][param]["timing"]["risk"]["mean"]
            table.add_row(
                alg,
                str(param),
                _prt(tot, None),
                _prt(pred, None) + f" ({_prt(pred/tot*100)}%)",
                _prt(cost, None) + f" ({_prt(cost/tot*100)}%)",
                _prt(risk, None, ".4f") + f" ({_prt(risk/tot*100, fmt='.3f')}%)",
            )
    console.print(table)
    print("")
    # Results Table
    kmax = {"f1_score", "precision", "recall", "accuracy"}
    best = (
        {k: max([stats[a][q][k] for a in stats for q in stats[a]]) for k in kmax}
        | {
            "alarm-to-collision": max(
                [
                    stats[a][q]["alarm-to-collision"]["mean"]
                    for a in stats
                    for q in stats[a]
                ]
            )
        }
        | {
            "timing": min(
                [
                    stats[a][q]["timing"]["total"]["mean"]
                    for a in stats
                    for q in stats[a]
                ]
            )
        }
    )
    # Performance Table
    table = Table(title="Results")
    table.add_column("Algorithm", justify="left", no_wrap=True)
    table.add_column("Param", justify="center")
    table.add_column("F1-Score", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("ATC", justify="center")
    table.add_column("Runtime", justify="center")
    for alg in sorted(stats.keys(), reverse=True):
        for param in sorted(stats[alg].keys(), reverse=True):
            table.add_row(
                alg,
                str(param),
                _prt(stats[alg][param]["f1_score"], best["f1_score"]),
                _prt(stats[alg][param]["precision"], best["precision"]),
                _prt(stats[alg][param]["recall"], best["recall"]),
                _prt(stats[alg][param]["accuracy"], best["accuracy"]),
                _prt(
                    stats[alg][param]["alarm-to-collision"]["mean"],
                    best["alarm-to-collision"],
                )
                + f" ({stats[alg][param]['alarm-to-collision']['median']:.2f})",
                _prt(stats[alg][param]["timing"]["total"]["mean"], best["timing"])
                + f" ({stats[alg][param]['timing']['total']['median']:.2f})",
            )
    print("")
    console.print(table)


def _generate_risk_plot(
    data, threshold, collision_step, fname, smoothing_sigma: Optional[float] = None
):
    if all([isinstance(x, SimpleRiskData) for x in data.values()]):
        _generate_copula_risk_plot(
            data, threshold, collision_step, fname, smoothing_sigma
        )
    else:
        _generate_bound_risk_plot(
            data, threshold, collision_step, fname, smoothing_sigma
        )


def _generate_copula_risk_plot(
    data, threshold, collision_step, fname, smoothing_sigma: Optional[float] = None
):
    df = pd.DataFrame.from_dict(
        {
            "Frame": [int(k) for k in data.keys()],
            "Risk": [v.risk for _, v in data.items()],
            "Confidence": [v.confidence for _, v in data.items()],
        }
    )
    if smoothing_sigma is not None:
        df["RiskSmooth"] = gaussian_filter1d(df.Risk, sigma=smoothing_sigma)
        df["ConfidenceSmooth"] = gaussian_filter1d(df.Confidence, sigma=smoothing_sigma)
        risk_label = "RiskSmooth"
        confidence_label = "ConfidenceSmooth"
    else:
        risk_label = "Risk"
        confidence_label = "Confidence"
    plt.figure()
    g = sns.lineplot(data=df, x="Frame", y=risk_label, color="blue")
    g = sns.lineplot(data=df, x="Frame", y=confidence_label, color="green")
    for th in threshold:
        g.axhline(th, color="red", linestyle="dashed")
    plt.xlabel("Time (frame)")
    plt.ylabel("Risk")
    plt.tight_layout()
    plt.savefig(fname, dpi=72)
    plt.close()


def _generate_bound_risk_plot(
    data, threshold, collision_step, fname, smoothing_sigma: Optional[float] = None
):
    df = pd.DataFrame.from_dict(
        {
            "Frame": [int(k) for k in data.keys()],
            "Lower": [k.upper for k in data.values()],
            "Upper": [k.lower for k in data.values()],
        }
    )
    if smoothing_sigma is not None:
        df["Lower"] = gaussian_filter1d(df.Lower, sigma=smoothing_sigma)
        df["Upper"] = gaussian_filter1d(df.Lower, sigma=smoothing_sigma)
    plt.figure()
    sns.lineplot(data=df, x="Frame", y="Lower", color="green")
    sns.lineplot(data=df, x="Frame", y="Upper", color="red")
    plt.fill_between(df.Frame, df.Lower, df.Upper, color="b", alpha=0.6)
    if collision_step is not None:
        plt.gca().axvline(collision_step, color="blue", linestyle="dashed")
    for th in threshold:
        plt.gca().axhline(th, color="red", linestyle="dashed")
    plt.tight_layout()
    plt.savefig(fname, dpi=72)
    plt.close()


def _generate_timing_plot(timing, fname, max_t=None):
    def _histplot(data, fname):
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        plt.figure()
        ax = sns.histplot(data=data, stat="percent")
        ax.set(xlabel="Runtime")
        plt.figtext(
            0.5,
            0.01,
            f"Mean: {mean:.2f} Median: {median:.2f} Std: {std:.2f}",
            ha="center",
            fontsize=12,
        )
        if max_t is not None:
            plt.xlim(-0.1, max_t)
        plt.tight_layout()
        plt.savefig(fname, dpi=72)

    _histplot(timing, fname)


def _generate_confusion_matrix_plot(cm, fname):
    plt.figure()
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    confusion_matrix = (cm.tn, cm.fp, cm.fn, cm.tp)
    group_counts = [f"{value:0.0f}" for value in confusion_matrix]
    group_percentages = [
        f"{value:.2%}" for value in confusion_matrix / np.sum(confusion_matrix)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    y = np.array(confusion_matrix).reshape((2, 2)) / np.sum(confusion_matrix)
    sns.heatmap(y, annot=labels, fmt="", cmap="Blues", cbar=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=72)


def generate_plots(
    cfg,
    alg,
    param,
    results,
    timing,
    experiment_folder,
    smoothing_sigma: Optional[float] = None,
):
    output_dir = experiment_folder / cfg.results.plots / f"{alg}_{param}"
    output_dir.mkdir(parents=True, exist_ok=True)
    risk = results.risk
    n_scenarios = len([scenario for log in risk for scenario in risk[log]])
    pbar = tqdm(total=n_scenarios + len(timing[alg][param]) + 1)
    # Scenario risk
    for log in risk:
        for scenario in risk[log]:
            fname = output_dir / f"{scenario}.pdf"
            cost_runtimes = results.experiment_results[log][
                scenario
            ].compute_cost_runtimes
            pred_runtime = results.experiment_results[log][scenario].prediction_runtimes
            risk_values = risk[log][scenario][alg][param]
            earliest_collision = results.experiment_results[log][
                scenario
            ].earliest_collision_time
            if isfinite(earliest_collision):
                delta = {
                    step: abs(t - earliest_collision)
                    for step, t in results.experiment_results[log][
                        scenario
                    ].timesteps_us.items()
                }
                j = [key for key, val in delta.items() if val == min(delta.values())][0]
            else:
                j = None
            assert set(cost_runtimes.keys()) == set(
                risk_values.keys()
            ), f"Number of steps does not match for alg {alg}"
            if alg == "ttc" or alg == "msd":
                th = cfg.risk.copula.threshold.thresholds
            elif alg == "cp":
                th = cfg.risk.cp.threshold.thresholds
            elif "hj" == alg:
                th = cfg.risk.hj.thresholds
            else:
                continue
                # raise ValueError(f"Unknown algorithm {alg}")
            _generate_risk_plot(risk_values, th, j, fname, smoothing_sigma)
            pbar.update(1)
    # Timing
    for elem in timing[alg][param]:
        fname = output_dir / f"timing_{elem}.pdf"
        _generate_timing_plot(timing[alg][param][elem], fname)
        pbar.update(1)

    # fname = output_dir / "total_timing.pdf"
    # _generate_timing_plot(tot_timing, fname, max_t=1.5)
    # pbar.update(1)
    # fname = output_dir / "prediction_timing.pdf"
    # _generate_timing_plot(prediction_runtimes, fname, max_t=1.5)
    # pbar.update(1)
    # fname = output_dir / "severity_estimation_timing.pdf"
    # _generate_timing_plot(sev_est_runtimes, fname, max_t=0.3)
    # pbar.update(1)
    # Confusion Matrix
    fname = output_dir / "confusion_matrix.pdf"
    if results.confusion_matrix is not None:
        _generate_confusion_matrix_plot(results.confusion_matrix[alg][param], fname)
    pbar.update(1)
    pbar.close()
    confusion_matrix_report(cfg, alg, param, results, experiment_folder)


def generate_roc_curve(cfg, results, experiment_folder):
    print("Generating ROC curve")
    output_dir = experiment_folder / cfg.results.plots
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_curve = dict()
    for alg, alg_results in results.confusion_matrix.items():
        tpr = []
        fpr = []
        for _, cm in alg_results.items():
            fpr.append(cm.fp / (cm.fp + cm.tn))
            tpr.append(cm.tp / (cm.tp + cm.fn))
        roc_curve[alg] = (fpr, tpr)

    # Plot
    fig, ax = plt.subplots()
    for alg, (fpr, tpr) in roc_curve.items():
        ax.plot(fpr, tpr, label=alg.upper())
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fname = output_dir / f"roc_curve.pdf"
    plt.savefig(fname, dpi=72)


def confusion_matrix_report(
    cfg, alg, param, results, experiment_folder: Optional[Path] = None
):
    confusion_matrix = {"tp": list(), "fp": list(), "tn": list(), "fn": list()}
    for log in results.experiment_results:
        for scenario in results.risk[log]:
            pred, gt, _ = classify(results, log, scenario, alg, param)
            if pred and gt:
                confusion_matrix["tp"].append(scenario)
            elif pred and not gt:
                confusion_matrix["fp"].append(scenario)
            elif not pred and gt:
                confusion_matrix["fn"].append(scenario)
            elif not pred and not gt:
                confusion_matrix["tn"].append(scenario)
    if experiment_folder is not None:
        output_dir = experiment_folder / cfg.results.plots / f"{alg}_{param}"
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / "confusion_matrix.yaml"
        OmegaConf.save(confusion_matrix, fname)
        print(f"Saved confusion matrix to {fname}")
    else:
        rprint("[bold red]Confusion Matrix[/bold red]")
        rprint(OmegaConf.to_yaml(confusion_matrix))


def find_all_algs(results):
    algs = set()
    risk = results.risk
    for log in risk:
        for scenario in risk[log]:
            for alg in risk[log][scenario]:
                for param in risk[log][scenario][alg]:
                    algs.add((alg, param))
            # all scenarios should be the same
            return algs


def start_nuboard(cfg, experiment_folder):
    print("Running NuBoard on http://localhost:5006")
    nuboard_file = experiment_folder / "nuboard_file.nuboard"
    assert nuboard_file.is_file(), "Could not find nuBoard file"
    print(f"Starting nuBoard with: {nuboard_file}")
    scenario_builder = NuPlanScenarioBuilder(
        data_root=cfg.nuplan.DATA_ROOT,
        map_root=cfg.nuplan.MAPS_ROOT,
        db_files=cfg.nuplan.DB_FILES,
        map_version=cfg.nuplan.MAP_VERSION,
    )
    nuboard = NuBoard(
        profiler_path=None,
        nuboard_paths=[str(nuboard_file)],
        scenario_builder=scenario_builder,
        port_number=5006,
        resource_prefix=None,
        vehicle_parameters=get_pacifica_parameters(),
    )
    nuboard.run()


def parse(fname):
    with open(fname, "rb") as f:
        results: PostprocessedResults = pickle.load(f)
    return results


def main_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument(
        "--plot",
        nargs=2,
        metavar=("alg", "param"),
        help="Generates the plot for ALG with PARAM",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Generates the plot for all algorithms and all parameters",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generates the summary for all algorithms",
    )
    parser.add_argument(
        "--nuboard",
        action="store_true",
        help="Starts nuBoard for scenario visualization",
    )
    parser.add_argument(
        "--confusion-matrix",
        nargs=2,
        metavar=("alg", "param"),
        help="Generate a yaml file with the scenario in each confusion matrix cell",
    )
    parser.add_argument("--roc", action="store_true", help="Generates the ROC curve")
    parser.add_argument(
        "--smoothing-sigma",
        "-s",
        type=float,
        help="Smoothing sigma for risk plots (default: None)",
    )
    args = parser.parse_args()

    experiment_folder = Path(args.experiment)

    if not experiment_folder.is_dir():
        experiment_folder = "experiments" / experiment_folder
    if not experiment_folder.is_dir():
        print(f"Could not find experiment folder: {experiment_folder}")
        exit()
    print("Using experiment folder:", experiment_folder)

    cfg_path = experiment_folder / "config.yaml"
    print("Using config:", cfg_path)
    cfg = OmegaConf.load(cfg_path)

    postprocessed_results_fname = experiment_folder / cfg.results.postprocessed_file
    results = None  # Lazy parsing...

    if args.plot_all:
        results = parse(postprocessed_results_fname) if results is None else results
        timing = _timing_info(cfg, results)
        all_algs = find_all_algs(results)
        for alg, param in all_algs:
            print(f"Generating plot for {alg} with param {param}...")
            generate_plots(
                cfg,
                alg,
                float(param),
                results,
                timing,
                experiment_folder,
                args.smoothing_sigma,
            )
    elif args.plot:
        results = parse(postprocessed_results_fname) if results is None else results
        timing = _timing_info(cfg, results)
        alg, param = args.plot
        generate_plots(
            cfg,
            alg,
            float(param),
            results,
            timing,
            experiment_folder,
            args.smoothing_sigma,
        )
    if args.roc:
        results = parse(postprocessed_results_fname) if results is None else results
        generate_roc_curve(cfg, results, experiment_folder)
    if args.summary:
        results = parse(postprocessed_results_fname) if results is None else results
        generate_summary(cfg, results, experiment_folder)
    if args.confusion_matrix:
        results = parse(postprocessed_results_fname) if results is None else results
        alg, param = args.confusion_matrix
        confusion_matrix_report(cfg, alg, float(param), results, None)
    if args.nuboard:
        start_nuboard(cfg, experiment_folder)
    print("Done!")


if __name__ == "__main__":
    main_app()
