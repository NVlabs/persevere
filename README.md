[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](https://github.com/NVlabs/perp-fail-severity/blob/main/LICENSE.md)
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# Perception Failure Severity Estimation

![Header](/docs/images/header.png)

This repository contains the code for [*"Task-Aware Risk Estimation of Perception Failures for Autonomous Vehicles"*](#), Proceedings of Robotics: Science and Systems (RSS), 2023.

## Requirements

- Python 3.9
- [Poetry >1.2](https://python-poetry.org/) (you can install it following [this guide](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions))

## Setup

### Setting up nuPlan

To run the code you need to install the [nuPlan dataset](https://www.nuscenes.org/nuplan).
At the moment only the mini version is required.
Follow [this guide](https://github.com/motional/nuplan-devkit/blob/master/docs/dataset_setup.md) to download and install the dataset.

This project relies on the environment variables to find the dataset.
In your `~/.bashrc` (or equivalent), add

```sh
export NUPLAN_DATA_ROOT="$HOME/datasets/nuplan/"
export NUPLAN_MAPS_ROOT="$HOME/datasets/nuplan/maps"
export NUPLAN_EXP_ROOT="$HOME/datasets/nuplan/exp"
```

replacing the path with the one you used.
the code expects `$NUPLAN_DATA_ROOT` to contain the `nuplan-v1.1` folder.

### Setting up the codebase

In a folder of your choice, run the following commands:

- Clone this project

```sh
git clone https://github.com/NVlabs/perp-fail-severity.git
```

- Clone and patch Trajectron++

```sh
git clone https://github.com/StanfordASL/Trajectron-plus-plus.git
cd Trajectron-plus-plus
git checkout 1031c7bd1a444273af378c1ec1dcca907ba59830
git apply ../perp-fail-severity/patches/trajectron.patch
cd ..
```

- Clone and patch NuPlan-devkit

```sh
git clone -b nuplan-devkit-v1.1 https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
git apply ../perp-fail-severity/patches/nuplan-v1.1.patch
cd ..
```

- Install all dependencies

```sh
cd perp-fail-severity
poetry install
```

This will set up the virtual environment with everything you need to run the code.

Note: by default this repository uses CPU-only JAX.
If you want to use the GPU version (which requires additional dependencies), comment L24 in `pyproject.toml` and and uncomment L25.
Then run `poetry install` or `poetry update` if you installed the dependencies already.

Before being able to run the code you must precompute the HJ-Reachability lookup table.
To do so, run

```sh
poetry run python3 precompute_value_function.py
```

## Run

### Configuration

The configuration is done through the `configs/general.yaml` file.
To read more about configuration read the [configuration guide](/docs/configuration.md).

### Run the code

To run the code, run

```sh
poetry run main
```

To visualize the scenario with nuBoard, run

```sh
poetry run viz experiments/2021-07-01_15-00-00 --nuboard
```

where `experiments/2021-07-01_15-00-00` is the path to the experiment folder.

You get generate a report with

```sh
poetry run viz experiments/2021-07-01_15-00-00 --summary
```

or generate all plots (risk, timing, and confusion matrices) with

```sh
poetry run viz experiments/2021-07-01_15-00-00 --plot-all
```


## Additional tasks

### Auto-formatter

This code is formatted with [Black](https://github.com/psf/black) (automatically installed).
You can format the code from your preferred IDE (e.g., VSCode), or, alternatively, you can run

```sh
poetry run poe format
```

to format all files in the project.

> **Please**: before pushing to git, make sure the project is properly formatted.

### Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com/) to make sure the code satisfies basic best practices.
To see/edit the hooks please read [pre-commit config file](.pre-commit-config.yaml).
When you clone the repository, the hooks are not automatically enabled, to do so run

```sh
poetry run pre-commit install
```

from now on, all the hooks will automatically execute every time you try to push.
If any of the hooks fail, the code will not be pushed.

If you want to manually run all pre-commit hooks on a repository, run `poetry run pre-commit run --all-files`.
This will check that staged and committed files are compliant.

## FAQ

- To use nuBoard from a remote machine you might need to setup port forwarding. On your machine run `ssh -NfL localhost:5006:localhost:5006 <user>@<remote>`.
- To omit debug log, run `export LOGURU_LEVEL=INFO` in the terminal session

## Citation

If you use this code or the dataset, please cite the following paper:

```
@article{antonante23rss-perceptionFailureSeverity,
  title={Task-Aware Risk Estimation of Perception Failures for Autonomous Vehicles},
  author={P. Antonante, S. Veer, K. Leung, X. Weng, L. Carlone and M. Pavone},
  year={2023}
}
```

## Licence

The source code is released under the [NSCL licence](https://github.com/NVlabs/perp-fail-severity/blob/main/LICENSE.md). The preprocessed dataset and pretrained models are under the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

