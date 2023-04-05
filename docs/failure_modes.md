# Failure Modes

## Placeholders

- `token` the track token of the vehicle that has the associated failure
- `offset` a named tuple `Offset(angle, distance)`
- `size` a named tuple `Size(width, length)`
- `noise` a named tuple `Noise(mean, std)`
- `object_type` the type of the object that is detected, i.e., one of the following:
  - `vehicle`
  - `pedestrian`
  - `bicycle`
  - `genericobject`
  - `traffic_cone`
  - `barrier`
- `failure` any failure mode generator (i.e., one supported failure)

## Supported Failures

> ### Misdetection
>
> **Signatures:** `Misdetection(token, offset:Offset, shape_ratio:Size, rotation:Angle, object_type)`
>
> **Effect**: Shrink the bounding box of the agent `token` by `shape_ratio`, shifts it by `offset` and rotate it by `rotation` radiants. It also changes the class to `object_type`.

> ### Missed Obstacle
> 
> **Signatures:** `MissedObstacle(token)`
> 
> **Effect**: The vehicle `token` is removed from detections.

> ### GhostObstacle
>
> **Signatures:** `GhostObstacle(offset:Offset, size:Size, object_type)`
>
> **Effect**: Create a ghost obstacle of class `object_type` that behaves as the ego-vehicle and offeset of `offset` wrt the ego-vehicle.

> ### TrafficLightMisdetection
>
> **Signatures:** `TrafficLightMisdetection(lane_id, traffic_light_state)`
>
> **Effect**: Switch the state of traffic light `lane_id` to `traffic_light_state`.

> ### Mislocalization
>
> **Signatures:** `Mislocalization(offset:Offset, rotation:Angle)`
>
> **Effect**: The ego-vehicle, and all detected obstacles, are mislocalized by `offset` and rotated by `rotation`.

> ### Flickering
>
> **Signatures:** `Flickering(failure, probability, duration)`
> 
> **Effect**: The failure `failure` (any other failure than Flickering) is applied with probability `probability`, and duration `duration`.

> ### AtTimestep
>
> **Signatures:** `AtTimestep(failure, time_us)`
>
> **Effect**: The failure `failure` (any other failure than AtTimestep) is applied from timestep `time_us` onward.
