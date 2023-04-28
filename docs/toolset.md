# Tools

The tool performs several useful operations on scenario configurations.

```
poetry run tools --help
Usage: tools [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  compare                     compare two scenario configurations
  compare-confusion-matrices  compare two confusion matrix reports
  extract                     extract scenarios from a configuration
  generate                    generate scenario configuration form scenariofilter
  info                        print information about a scenario configuration
  ls                          list scenarios in a configuration file
  merge                       merge two scenario configurations
  random                      select N random scenarios from a scenario configuration
  types                       list scenario types
  validate                    validate a scenario configuration
```

To explore the options of a command, run `poetry run tools COMMAND --help`.

## Scenario genration

The tool `generate` allows to generate a scenario configuration from a scenario filter.
A scenario filter is a yaml file containing a one or more of the following `scenario_types`, `scenario_tokens`, `num_scenarios_per_type`, `limit_total_scenarios`.

```yaml
scenario_types:
  - traversing_intersection
  - near_long_vehicle
  - medium_magnitude_speed
  - traversing_traffic_light_intersection
  - high_magnitude_speed
  - stationary_at_traffic_light_without_lead
  - on_traffic_light_intersection
  - near_multiple_vehicles
  - starting_protected_cross_turn
  - near_pedestrian_on_crosswalk
scenario_tokens:
num_scenarios_per_type: 10
limit_total_scenarios:
```

For example to generate a scenario configuration, run

```sh
poetry run tools generate -f configs/scenario_filter.yaml configs/nuplan/default.yaml configs/scenarios/generated.yaml
```
