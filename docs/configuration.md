# Configuration

The configuration is managed by the file `configs/general.yaml`.
The file is divided in sections, the most important are described below.

### Scenarios

The section `scenarios` (under `defaults`) contains the list of scenarios to be run.
The configuration manager [Hydra](https://github.com/facebookresearch/hydra) automatically load the scenarios file from the folder `/configs/scenarios/`.

At minimum, a scenario file must contain a list of scenarios, each specified by its name the `name` (nuPlans scenario name).
Optionally a scenario can specify the failure modes to be injected in the scenario.
The failure modes are specified as a list under the key `failures`, for example:

```yaml
- type: starting_straight_stop_sign_intersection_traversal
  log: 2021.06.07.12.54.00_veh-35_01843_02314
  name: d7dce7247a0e57f9
  failures:
    - Flickering(MissedObstacle('440a808d665c5eb1'))
```

here the scenario `d7dce7247a0e57f9` is subject to a flickering missed obstacle.
The list of supported failure modes is described in the [failure modes documentation](/docs/failure_modes.md).

To create a new list of scenarios add yaml file in the folder `/configs/scenarios/` and add the name of the file in the `configs/config.yaml` file (under `scenarios`).
To help the management of a scenario file you can use the associated [toolset](/docs/toolset.md).

## Plausible Scene Generator (`hypothesis_generator`)

The section `hypothesis_generator` (under `defaults`) contains the configuration for the plausible scene generator.
Similarly to the scenarios, the configuration manager loads a file from the folder `/configs/hypothesis_generators/`.

The Plausible Scene Generator (here called hypothesis generator) supports 3 modes:
- Ground truth: the ground truth scene is used as the plausible scene.
- Noisy: perturbs the ground truth with noise
- Fixed: the unkown values (e.g., velocity, position) are fixed to a constant value.

Refer to the [examples](/configs/hypothesis_generator) to see how to configure the plausible scene generator.

## Risk Assessment (`risk`)

The risk assessment is configured under the section `risk` (under `defaults`).
This section contains the configuration for the copula estimation, the HJ-Reachability, and the collision probability baselines.

### Copula estimation (`copula`)

There are two copula estimation methods:
- PAC-Bounds (`type: bound`)
- Direct Copula Estimation (`type: copula`)

In the case of PAC-Bounds, we need to specify the risk aversion, risk thresholds and the confindence (1-alpha).

```yaml
copula:
  type: bound
  risk_aversion: [0.9, 0.95, 0.99]
  confidence: 0.9
  threshold:
    thresholds: [0.4, 0.8, 0.99]
    labels: ["low", "medium", "high", "critical"]
```

This configuration will compute the risk for the risk aversion 0.9, 0.95, and 0.99, with a confidence of 0.9.
If the p-RSR value is below 40%, the scenario risk is classified as `low` risk, if it's between 40% and 80% the scenario risk is classified as `medium`, if it's between 80% and 99% the scenario risk is `high`, and if it's above 99% the scenario risk is `critical`.

In the case of Direct Copula Estimation, we need can omit the confidence value, and specify the risk threshold.
At the moment just one risk threshold is supported.
Tis is not a limitation of the approach but of the implementation (easily fixable).

```yaml
copula:
  type: copula
  risk_aversion: [0.9, 0.95, 0.99]
  threshold: 0.99
```

### HJ-Reachability (`hj`)

It requires just one parameter, the the threshold that defines the risky scene.
It is input as a list of thresholds, each value is the threshold for a different risk aversion.

```yaml
hj:
  thresholds: [0 , -1.5, -3]
```

### Collision Probability Baselines (`cp`)

Similarly to the copula, it requires the risk aversion parameter and the risk thresholds.

```yaml
cp:
  risk_aversion: [0.9, 0.95, 0.99]
  threshold:
    thresholds: [0.4, 0.8]
    labels: ["low", "medium", "high" ]
```

## Others

### Trajectron++ (`trajectron`)

The section `trajectron` contains the configuration for the Trajectron++ model.
The `model` is the path to the pre-trained model (shipped with the code).
`num_samples` is the sample size for each scene, and the `subsample_ratio` is the ratio of the trajectory to be inputed into the model 
Finally, `dt` is the time step of the model.

```yaml
trajectron:
  dt: 0.5
  model: "models/trajectron"
  subsample_ratio: 0.1
  num_samples: 1000
  scene_radius: 100.0
```

### HJ-Reachability (`hj_reachability`)

The section `hj_reachability` contains the configuration for the HJ-Reachability model.
The `lut_table` is the path to the precomputed lookup table (see the [readme](README.md) for setup instructions).
The `scene_radius` is the radius of the scene to be considered by the HJ-Reachability model.

```yaml
hj_reachability:
  lut_table: "models/hj/hj_reachability.pkl"
  scene_radius: 100.0
```

### Planner (`planner`)

The section `planner` contains the configuration for the agents and ego planners.
Both the ego vehicle and the agents use the IDM planner, with the same parameters specified in this section.

```yaml
planner:
  IDMPlanner:
    target_velocity: 10
    min_gap_to_lead_agent: 0.5
    headway_time: 1.5
    accel_max: 1.0
    decel_max: 2.0
    planned_trajectory_samples: 10
    planned_trajectory_sample_interval: 0.2
    occupancy_map_radius: 20
  PredictionRiskPlanner:
    verbose: False
```
