import ast
import re
from collections import ChainMap
from random import shuffle
from typing import Set

import click
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from severity_estimation.fault_injection.common_failures import (
    COMMON_FAILURES,
    TEMPORAL_FAILURES,
)

BASE_FAILURES = sorted(list({c.__name__ for c in COMMON_FAILURES}))
TEMPORAL_TYPES = sorted(list({c.__name__ for c in TEMPORAL_FAILURES}))
SCENARIO_NAME_FORMAT = re.compile(r"^[a-fA-F0-9]{16}$")


def _get_function_names(code) -> Set[str]:
    tree = ast.parse(code)
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)}


def _get_scenarios(nuplan_cfg, filter_cfg=None):
    # fmt: off
    filter = ScenarioFilter(
        scenario_types=filter_cfg.scenario_types if filter_cfg else None,
        scenario_tokens=filter_cfg.scenario_tokens if filter_cfg else None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=filter_cfg.num_scenarios_per_type if filter_cfg else None,
        limit_total_scenarios=filter_cfg.limit_total_scenarios if filter_cfg else None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        timestamp_threshold_s=None,
    )
    # fmt: on
    scenario_builder = NuPlanScenarioBuilder(
        data_root=nuplan_cfg.DATA_ROOT,
        map_root=nuplan_cfg.MAPS_ROOT,
        db_files=nuplan_cfg.DB_FILES,
        map_version=nuplan_cfg.MAP_VERSION,
    )
    scenarios = scenario_builder.get_scenarios(filter, Sequential())
    return scenarios


def validate_scenario_config(cfg, check_unique=True):
    unique_names = set()
    for scenario in cfg:
        assert isinstance(scenario, DictConfig), "each scenario must be a dict"
        assert "type" in scenario, "scenario must have a type"
        assert "log" in scenario, "scenario must have a log name"
        assert "name" in scenario, "scenario must have a name"
        assert SCENARIO_NAME_FORMAT.fullmatch(
            scenario.name
        ), f"invalid scenario name `{scenario.name}`"
        assert "failures" in scenario, "scenario must have failures"
        if check_unique:
            assert (
                scenario.name not in unique_names
            ), f"duplicate scenario name {scenario.name}"
            unique_names.add(scenario.name)
        if scenario.failures is not None:
            assert isinstance(scenario.failures, ListConfig), "failures must be a list"
            for failure in scenario.failures:
                assert isinstance(failure, str), "failure must be a string"
                assert any(
                    f in failure for f in BASE_FAILURES
                ), f"unknown failure {failure}"


def _load_and_validate_config(config, check_unique=True):
    cfg = OmegaConf.load(config)
    if isinstance(cfg, DictConfig):
        assert "scenarios" in cfg, "must be a ListConfig or have the scenarios key"
        cfg = cfg.scenarios
    assert isinstance(cfg, ListConfig), "scenarios must be a list"
    validate_scenario_config(cfg)
    return cfg


@click.group()
def cli():
    pass


@cli.command(short_help="print information about a scenario configuration")
@click.argument("config", type=click.Path(exists=True))
def info(config):
    """Print information about a scenario configuration.
    \n
    CONFIG is the scenario configuration to print information about.
    """

    FailureActivation = lambda: {k: 0 for k in TEMPORAL_TYPES} | {"Static": 0}

    def find_element(s, string):
        for element in s:
            if element in string:
                return element
        return None

    cfg = _load_and_validate_config(config)
    # Failures Summary
    summary = {k: FailureActivation() for k in BASE_FAILURES}
    console = Console()
    num_scenarios = len(cfg)
    num_failures = 0
    for scenario in cfg:
        if scenario.failures is None:
            continue
        for failure in scenario.failures:
            num_failures += 1
            fcn = _get_function_names(failure)
            for f in BASE_FAILURES:
                if f in fcn:
                    temporal_found = False
                    for t in TEMPORAL_TYPES:
                        if t in fcn:
                            summary[f][t] += 1
                            temporal_found = True
                    summary[f]["Static"] += 1 if not temporal_found else 0
    table = Table(title=f"Failures Analytics (scenarios {num_scenarios})")
    table.add_column("Failure", justify="left", no_wrap=True)
    table.add_column("Total", justify="center", no_wrap=True)
    table.add_column("Static", justify="center", no_wrap=True)
    for k in TEMPORAL_TYPES:
        table.add_column(k, justify="center", no_wrap=True)
    for f, v in sorted(summary.items()):
        tot = sum(v.values())
        tot_str = f"{tot/num_failures*100:.2f}% ({tot})" if tot > 0 else None
        z = [f, tot_str] + [
            f"{v[t]/num_failures*100:.2f}% ({v[t]})" if v[t] > 0 else None
            for t in ["Static"] + TEMPORAL_TYPES
        ]
        table.add_row(*z)
    console.print(table)
    # Type Summary
    summary = dict()
    for scenario in cfg:
        if scenario.type not in summary:
            summary[scenario.type] = 0
        summary[scenario.type] += 1
    table = Table(title=f"Scenario Types Analytics (scenarios {num_scenarios})")
    table.add_column("Type", justify="left", no_wrap=True)
    table.add_column("Percentage", justify="center", no_wrap=True)
    table.add_column("Count", justify="center", no_wrap=True)
    for t, v in sorted(summary.items()):
        table.add_row(t, f"{v/num_scenarios*100:.2f}%", str(v))
    console.print(table)
    # Subtype Summary
    if any("subtype" in s for s in cfg):
        summary = dict()
        for scenario in cfg:
            st = scenario.subtype if "subtype" in scenario else "Unknown"
            if st not in summary:
                summary[st] = {"static": 0, "dynamic": 0}
            if any("Flickering" in f for f in scenario.failures):
                summary[st]["dynamic"] += 1
            else:
                summary[st]["static"] += 1
        table = Table(title=f"Scenario Subtypes Analytics (scenarios {num_scenarios})")
        table.add_column("Subtype", justify="left", no_wrap=True)
        table.add_column("Static", justify="center", no_wrap=True)
        table.add_column("Dynamic", justify="center", no_wrap=True)
        # table.add_column("Percentage", justify="center", no_wrap=True)
        # table.add_column("Count", justify="center", no_wrap=True)
        for t, v in sorted(summary.items()):
            table.add_row(
                t,
                str(v["static"]) if v["static"] > 0 else None,
                str(v["dynamic"]) if v["dynamic"] > 0 else None,
            )
            # table.add_row(t, f"{v/num_scenarios*100:.2f}%", str(v))
        console.print(table)


@cli.command(short_help="list scenarios in a configuration file")
@click.argument("config", type=click.Path(exists=True))
def ls(config):
    """List scenarios in a configuration file.
    \n
    CONFIG is the scenario configuration file to list scenarios from.
    """
    cfg = _load_and_validate_config(config)
    names = {s.name for s in cfg}
    [print(n) for n in sorted(names)]
    print(f"Total: {len(names)} scenarios")


@cli.command(short_help="validate a scenario configuration")
@click.argument("config", type=click.Path(exists=True))
def validate(config):
    """Validate a scenario configuration.
    \n
    CONFIG is the scenario configuration to validate.
    """
    try:
        cfg = _load_and_validate_config(config)
    except AssertionError as e:
        print(f"Scenario configuration is invalid:\nError: `{e}`")
        exit(1)
    else:
        print(f"Num scenarios: {len(cfg)}")
        print("Scenario configuration is valid")
    if not all(s.failures is not None for s in cfg):
        x = {s.name for s in cfg if s.failures is None}
        print("WARNING: Some scenarios do not have failures defined")
        print(f" ↳ {x}")
    if any("subtype" in s for s in cfg):
        if not all("subtype" in s for s in cfg):
            x = {s.name for s in cfg if "subtype" not in s}
            print("WARNING: Some scenarios do not have a subtype defined")
            print(f" ↳ {x}")


@cli.command(short_help="validate a scenario configuration")
@click.argument("config", type=click.Path(exists=True))
@click.argument("n", type=int)
def random(config, n):
    """Get N random scenarios from a scenario configuration.
    \n
    CONFIG is the scenario configuration to validate.
    \n
    N is the number of scenarios to get.
    """
    cfg = _load_and_validate_config(config)
    if n > len(cfg):
        print(
            f"ERROR: Cannot get {n} random scenarios from a configuration with {len(cfg)} scenarios"
        )
        exit(1)
    shuffle(cfg)
    print(OmegaConf.to_yaml(cfg[:n]))


@cli.command(short_help="compare two scenario configurations")
@click.argument("config1", type=click.Path(exists=True))
@click.argument("config2", type=click.Path(exists=True))
def compare(config1, config2):
    """Compare two scenario configurations.
    \n
    CONFIG1 is the first scenario configuration to compare.
    \n
    CONFIG2 is the second scenario configuration to compare.
    """
    cfg1 = _load_and_validate_config(config1)
    cfg2 = _load_and_validate_config(config2)
    scenarios_1 = {s.name for s in cfg1}
    scenarios_2 = {s.name for s in cfg2}
    cfg1_unique = scenarios_1 - scenarios_2
    cfg2_unique = scenarios_2 - scenarios_1
    both = scenarios_1 & scenarios_2
    print("-" * 80)
    print(f"Scenarios UNIQUE in {config1} ({len(cfg1_unique)}/{len(scenarios_1)})")
    print("-" * 80)
    [print(n) for n in sorted(cfg1_unique)]
    print("-" * 80)
    print(f"Scenarios UNIQUE in {config2} ({len(cfg2_unique)}/{len(scenarios_2)})")
    print("-" * 80)
    [print(n) for n in sorted(cfg2_unique)]
    print("-" * 80)
    print(f"Scenarios in BOTH configs ({len(both)})")
    print("-" * 80)
    [print(n) for n in sorted(both)]


@cli.command(short_help="extract scenarios from a configuration")
@click.argument("config", type=click.Path(exists=True))
@click.argument("scenarios", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
def extract(config, scenarios, output):
    """Extract scenarios from a configuration.

    CONFIG is the scenario configuration to extract from.
    \n
    SCENARIOS is the list of scenarios to extract, one for each line.
    \n
    OUTPUT is the path to the output file.
    """
    assert config != output, "Cannot overwrite input file"
    desired_scenarios = dict()
    scenario_names = OmegaConf.load(scenarios)
    assert isinstance(scenario_names, ListConfig), "SCENARIOS must be a list"
    # with open(scenarios) as fp:
    #     scenario_names = fp.read().splitlines()
    for name in scenario_names:
        assert SCENARIO_NAME_FORMAT.fullmatch(name), f"Invalid scenario name `{name}`"
        if name not in desired_scenarios:
            desired_scenarios[name] = False
        else:
            print(f"Duplicate scenario name: {name}")
    if len(desired_scenarios) == 0:
        print("No scenarios to extract")
        return
    cfg = _load_and_validate_config(config)
    extracted = []
    for scenario in cfg:
        if scenario.name in desired_scenarios:
            extracted.append(scenario)
            desired_scenarios[scenario.name] = True
    for name, found in desired_scenarios.items():
        if not found:
            print(f"Scenario {name} not found")
    OmegaConf.save(OmegaConf.create(extracted), output)
    print(f"Saved {len(extracted)} scenarios to {output}")


@cli.command(short_help="merge two scenario configurations")
@click.argument("config1", type=click.Path(exists=True))
@click.argument("config2", type=click.Path(exists=True))
@click.argument("merge", type=str)
@click.argument("output", type=click.Path(exists=False))
def merge(config1, config2, merge, output):
    """Merge two scenario configurations.

    CONFIG1 and CONFIG2 are the two scenario configurations to merge.
    \n
    MERGE is the merge strategy to use. It can be one of the following:\n
        - `left`: use the scenarios from CONFIG1\n
        - `right`: use the scenarios from CONFIG2 \n
        - `unique`: use the scenarios from CONFIG1 and CONFIG2 that are not in the other
    \n
    OUTPUT is the path to the output file.
    """
    cfg1 = _load_and_validate_config(config1)
    cfg2 = _load_and_validate_config(config2)
    if merge == "left" or merge == "unique":
        scenarios = cfg1
        seek = cfg2
    elif merge == "right":
        scenarios = cfg2
        seek = cfg1
    else:
        raise ValueError(f"Invalid merge strategy {merge}")
    for scenario in seek:
        if scenario.name not in {s.name for s in scenarios}:
            scenarios.append(scenario)
        else:
            if merge == "unique":
                print(f"Duplicate scenario name: {scenario.name}, skipping.")
                # Remove the scenario scenarios
                scenarios = [s for s in scenarios if s.name != scenario.name]
            else:
                print(f"Duplicate scenario name: {scenario.name}, keeping {merge}.")
    OmegaConf.save(scenarios, output)
    print(f"Saved {len(scenarios)} scenarios to {output}")


@cli.command(short_help="generate scenario configuration form scenario filter")
@click.argument("nuplan", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
@click.option(
    "-f",
    "--filter",
    help="Filter configuration file if None, all scenarios are generated.",
    type=click.Path(exists=True),
)
def generate(nuplan, filter, output):
    """Generate a scenario configuration.

    NUPLAN is the path to the NuPlan configuration file.
    \n
    OUTPUT is the path to the output file.
    """
    filter_cfg = OmegaConf.load(filter) if filter is not None else None
    scenarios = _get_scenarios(OmegaConf.load(nuplan), filter_cfg)
    config_content = []
    for scenario in scenarios:
        config_content.append(
            {
                "type": scenario.scenario_type,
                "log": scenario.log_name,
                "name": scenario.scenario_name,
                "failures": None,
            }
        )
    OmegaConf.save(OmegaConf.create(config_content), output)
    print(f"Saved {len(scenarios)} scenarios to {output}")


@cli.command(short_help="list scenario types")
@click.argument("nuplan", type=click.Path(exists=True))
@click.option(
    "-c",
    "--config",
    help="Scenario configuration file, if None, all scenarios are considered.",
    type=click.Path(exists=True),
)
def types(nuplan, config):
    """Lists all types of scenarios.

    NUPLAN is the path to the NuPlan configuration file.
    """
    if config is None:
        scenarios = _get_scenarios(OmegaConf.load(nuplan), None)
        scenario_types = {s.scenario_type for s in scenarios}
    else:
        cfg = _load_and_validate_config(config)
        scenario_types = {s.type for s in cfg}
    [print(t) for t in sorted(scenario_types)]


@cli.command(short_help="compare two confusion matrix reports")
@click.argument("matrix1", type=click.Path(exists=True))
@click.argument("matrix2", type=click.Path(exists=True))
def compare_confusion_matrices(matrix1, matrix2):
    """Compare two confusion matrix reports, indicating the differences.

    MATRIX1 is the path to first confusion matrix [.yaml].
    \n
    MATRIX2 is the path to second confusion matrix [.yaml].
    """
    print("Comparing confusion matrices")
    print(f"Matrix 1: {matrix1}")
    print(f"Matrix 2: {matrix2}\n")
    assert matrix1.endswith(".yaml") and matrix2.endswith(".yaml"), "Invalid file type"
    matrix1 = OmegaConf.load(matrix1)
    assert {"tp", "tn", "fp", "fn"} == set(matrix1.keys()), "Invalid matrix 1"
    matrix2 = OmegaConf.load(matrix2)
    assert {"tp", "tn", "fp", "fn"} == set(matrix2.keys()), "Invalid matrix 2"
    # Flip dictionaries i.e. {scenario: class}
    scenarios_1 = dict(ChainMap(*[{x: k for x in v} for k, v in matrix1.items()]))
    scenarios_2 = dict(ChainMap(*[{x: k for x in v} for k, v in matrix2.items()]))
    assert (
        scenarios_1.keys() == scenarios_2.keys()
    ), "The two matrices have different scenarios"

    diff = 0
    for scenario in scenarios_1.keys():
        a = scenarios_1[scenario]
        b = scenarios_2[scenario]
        if a != b:
            if a in {"tp", "tn"} and b in {"tp", "tn"}:
                c = "blue"
            elif a in {"fp", "fn"} and b in {"tp", "tn"}:
                c = "green"
            else:
                c = "red"
            rprint(f"[bold]{scenario}[/bold]: [{c}]{a} -> {b}[/{c}]")
            diff += 1
    if diff > 0:
        print(f"\nTotal differences: {diff}")
    else:
        print("No differences found")
