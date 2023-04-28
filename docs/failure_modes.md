# Failure Modes

## Datatypes

**Offset**: Relative offset in polar coordinates
- `angle` in radians
- `distance` in meters
- `noise_std` 

**Position**: Absolute position in the map
- `x` in meters
- `y` in meters
- `heading` in radians
- `noise_std`

**Size**: Size of the object
- `width` in meters
- `length` in meters
- `noise_std`

**Number**: Scalar number, it can be
- `Gaussian`: Gaussian distribution with mean `mean` and standard deviation `std`
- `Uniform`: Uniform distribution between `min` and `max`
- `Constant`: Constant value `value`

**Angle**: Angle in radians
- `angle` in radians (of type number)

**ObjectType**: A string representing an object class:
- `vehicle`
- `pedestrian`
- `bicycle`
- `genericobject`
- `traffic_cone`
- `barrier`

**Token**: A string representing the unique track token of an object

## Supported Failure Modes

> ### Missed Obstacle
> 
> **Signatures:** `MissedObstacle(token:Token)`
> 
> **Effect**: The vehicle `token` is removed from detections.

> ### GhostObstacle
>
> **Signatures:** `GhostObstacle(offset:Offset|Position, rotation:Angle, size:Size, velocity_ratio: Number, object_type:ObjectType)`
>
> **Effect**: Injects a ghost obstacle of class `object_type`. The offset is w.r.t the ego-vehicle. Rotation introduces an additive noise to the ego-vehicle heading. Size replaces the object size. Velocity ratio scales the object velocity.

> ### Misdetection
>
> **Signatures:** `Misdetection(token:Token, offset:Offset, shape_ratio:Size, rotation:Angle, velocity_ratio: Number, object_type:ObjectType)`
>
> **Effect**: A generic failure for misdetections (i.e., misposition, wrong velocity/orientation/size etc.) The failure only affects the object with token `token`. Offset moves the object relative to its grount truth location. `shape_ratio` scales the object size. Rotation introduces an additive noise to the object heading. Velocity ratio scales the object velocity. `object_type` replaces the object class (i.e., misclassification).

> ### TrafficLightMisdetection
>
> **Signatures:** `TrafficLightMisdetection(selector:str, traffic_light_state)`
>
> **Effect**: Impose the traffic light state `traffic_light_state` to the traffic light with selector `selector`. The selector can be `all` if all traffic lights should be affected, `proximal` if only the closes to the ego-vehicle should be affected.

> ### Mislocalization
>
> **Signatures:** `Mislocalization(offset:Offset, rotation:Angle)`
>
> **Effect**: The ego-vehicle, and all detected obstacles, are mislocalized by `offset` and rotated by `rotation`.

> ### Flickering
>
> **Signatures:** `Flickering(failure:FailureGenerator, probability:float, duration:float)`
> 
> **Effect**: The failure `failure` (any other failure than `Flickering` or `AtTimestep`) is applied with probability `probability`, and duration `duration`.

> ### AtTimestep
>
> **Signatures:** `AtTimestep(failure:FailureGenerator, time_us:int, stop_at:int=Infinity)`
>
> **Effect**: The failure `failure` (any other failure than `AtTimestep`) is applied from timestep `time_us` until `stop_at` (if finite).

For examples of how to use these failures, see the [scenario configuration](/configs/scenarios/handpicked_100.yaml).
