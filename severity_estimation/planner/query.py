class Query:
    class q:
        def __init__(self, options):
            self.name = options[0]
            self.options = options

        def __str__(self):
            return self.name

        def __call__(self, i):
            return self.options[i]

    dt = 0.5  # TODO[antonap]: maybe it should be a parameter
    position = q(("position", {"position": ("x", "y")}))
    velocity = q(("velocity", {"velocity": ("x", "y")}))
    velocity_norm = q(("velocity_norm", {"velocity": ("norm",)}))
    acceleration = q(("acceleration", {"acceleration": ("x", "y")}))
    acceleration_norm = q(("acceleration_norm", {"acceleration": ("norm",)}))
    jerk = q(("jerk", (acceleration,)))
    jerk_norm = q(("jerk_norm", (jerk,)))

    heading = q(("heading", {"heading": ("°",)}))
    heading_rate = q(("heading_rate", (heading,)))
    heading_acceleration = q(("heading_acceleration", (heading_rate,)))
    rotation_matrix = q(("rotation_matrix", (heading,)))

    rotated_position = q(("rotated_position", (rotation_matrix, position)))
    rotated_velocity = q(("rotated_velocity", (rotation_matrix, velocity)))
    rotated_acceleration = q(("rotated_acceleration", (rotation_matrix, acceleration)))
    rotated_jerk = q(("rotated_jerk", (rotation_matrix, jerk)))

    lon_velocity = q(("lon_velocity", (rotated_velocity,), 0))
    lat_velocity = q(("lat_velocity", (rotated_velocity,), 1))
    lon_acceleration = q(("lon_acceleration", (rotated_acceleration,), 0))
    lat_acceleration = q(("lat_acceleration", (rotated_acceleration,), 1))
    lon_jerk = q(("lon_jerk", (rotated_jerk,), 0))
    lat_jerk = q(("lat_jerk", (rotated_jerk,), 1))

    # For Predictions
    true_position = q(("true_position", (position,)))
    true_velocity = q(("true_velocity", (velocity,)))
    weighted_mean = q(("weighted_mean", ()))
    mean_distance_error = q(("mean_distance_error", (true_position, weighted_mean)))
    position_likelihood = q(("position_likelihood", (true_position, weighted_mean)))

    # For 2 nodes (name, sub query, ego query, agent query)
    relative_position = q(("relative_position", (), (position,), (position,)))
    relative_velocity = q(("relative_velocity", (), (velocity,), (velocity,)))
    relative_acceleration = q(
        ("relative_acceleration", (), (acceleration,), (acceleration,))
    )
    relative_heading = q(("relative_heading", (), (heading,), (heading,)))
    true_relative_position = q(
        ("true_relative_position", (), (position,), (true_position,))
    )
    true_relative_velocity = q(
        ("true_relative_velocity", (), (velocity,), (true_velocity,))
    )
    rotated_relative_position = q(
        ("rotated_relative_position", (relative_position,), (rotation_matrix,), ())
    )
    rotated_relative_velocity = q(
        ("rotated_relative_velocity", (relative_velocity,), (rotation_matrix,), ())
    )
    rotated_relative_acceleration = q(
        (
            "rotated_relative_acceleration",
            (relative_acceleration,),
            (rotation_matrix,),
            (),
        )
    )
