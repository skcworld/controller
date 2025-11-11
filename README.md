# Controller Package

Python implementation of MAP (Model-Adaptive Pursuit) and PP (Pure Pursuit) controllers for autonomous racing, ported from the `crazy_controller` C++ ROS2 package.

## Overview

This package provides two control algorithms:

- **MAP Controller**: L1 adaptive control with acceleration-aware steering and startup speed blending
- **PP Controller**: Geometric Pure Pursuit path tracking with adaptive lookahead distance

Both controllers support:
- Dynamic parameter reconfiguration
- Steering lookup tables (MAP only uses table, PP uses geometric formula)
- Lateral error-based speed adjustment
- Acceleration-based steering scaling
- TF2-based localization

## Package Structure

```
controller/
├── controller/
│   ├── __init__.py
│   ├── controller_manager.py   # Unified ROS2 node (MAP/PP mode selection)
│   ├── map.py                   # MAP controller algorithm
│   ├── pp.py                    # PP controller algorithm
│   └── utils/
│       ├── __init__.py
│       ├── steering_lookup.py   # CSV-based steering angle lookup
│       └── parameter_handler.py # Dynamic parameter event handling
├── config/
│   ├── l1_params.yaml           # MAP controller parameters
│   ├── l1_params_sim.yaml       # MAP controller simulation parameters
│   ├── pp_params.yaml           # PP controller parameters
│   ├── NUC4_pacejka_lookup_table.csv  # Steering lookup table (real vehicle)
│   ├── RBC1_pacejka_lookup_table.csv  # Steering lookup table (alternative)
│   └── SIM_linear_lookup_table.csv    # Steering lookup table (simulation)
├── launch/
│   └── controller_launch.py     # Launch file with mode selection
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## Installation

1. Navigate to your ROS2 workspace:
```bash
cd ~/ros2_ws/src
```

2. Copy or clone this package into the workspace

3. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select controller
source install/setup.bash
```

## Usage

### Launch Commands

The controller requires two arguments:
- `mode`: Controller algorithm (`MAP` or `PP`)
- `env`: Environment (`real` or `sim`)

**Examples:**

```bash
# Real vehicle with MAP controller (default)
ros2 launch controller controller_launch.py mode:=MAP env:=real

# Simulation with MAP controller
ros2 launch controller controller_launch.py mode:=MAP env:=sim

# Real vehicle with PP controller
ros2 launch controller controller_launch.py mode:=PP env:=real

# Simulation with PP controller
ros2 launch controller controller_launch.py mode:=PP env:=sim
```

**Default values:**
- `mode`: `MAP`
- `env`: `real`

So you can also use:
```bash
# Same as mode:=MAP env:=real
ros2 launch controller controller_launch.py

# Same as mode:=MAP env:=sim
ros2 launch controller controller_launch.py env:=sim
```

### Parameter and Lookup Table Mapping

| mode | env  | Parameter File       | Lookup Table                  |
|------|------|---------------------|-------------------------------|
| MAP  | real | l1_params.yaml      | NUC4_pacejka_lookup_table.csv |
| MAP  | sim  | l1_params_sim.yaml  | SIM_linear_lookup_table.csv   |
| PP   | real | pp_params.yaml      | N/A (uses geometric formula)  |
| PP   | sim  | pp_params_sim.yaml  | N/A (uses geometric formula)  |

**Note:** PP controller uses Pure Pursuit geometric formula and does not require a steering lookup table.

### Required Topics

The controller subscribes to:
- `/local_waypoints` (ae_hyu_msgs/WpntArray): Local waypoint array with [x, y, vx, d_left, d_right, s, kappa, psi, ax]
- `/odom` (nav_msgs/Odometry): Vehicle odometry (velocity only, position from TF)
- `/frenet/odom` (nav_msgs/Odometry): Frenet frame coordinates [s, d, vs, vd]
- `/sensors/imu/raw` (sensor_msgs/Imu): IMU data for acceleration

The controller publishes to:
- `/drive` (ackermann_msgs/AckermannDriveStamped): Steering and speed commands

### TF Frames

The controller uses TF2 to get vehicle pose:
- Source frame: `map` (configurable via `map_frame` parameter)
- Target frame: `base_link` (configurable via `base_link_frame` parameter)

## Parameters

### Common Parameters (MAP and PP)

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| `mode` | string | Controller mode: "MAP" or "PP" | - |
| `lookup_table_path` | string | Path to steering lookup CSV | - |
| `map_frame` | string | TF source frame | - |
| `base_link_frame` | string | TF target frame | - |
| `t_clip_min` | double | Minimum L1 lookahead distance [m] | 0.0 - 1.5 |
| `t_clip_max` | double | Maximum L1 lookahead distance [m] | 0.0 - 10.0 |
| `m_l1` | double | L1 distance velocity multiplier [s] | 0.0 - 1.0 |
| `q_l1` | double | L1 distance base offset [m] | -1.0 - 1.0 |
| `speed_lookahead` | double | Lookahead time for speed command [s] | 0.0 - 1.0 |
| `lat_err_coeff` | double | Lateral error speed reduction coefficient | 0.0 - 1.0 |
| `acc_scaler_for_steer` | double | Steering scale during acceleration | 0.0 - 1.5 |
| `dec_scaler_for_steer` | double | Steering scale during deceleration | 0.0 - 1.5 |
| `start_scale_speed` | double | Speed at which steering downscaling begins [m/s] | 0.0 - 10.0 |
| `end_scale_speed` | double | Speed at which steering downscaling reaches max [m/s] | 0.0 - 10.0 |
| `downscale_factor` | double | Maximum steering downscale factor | 0.0 - 0.5 |
| `speed_lookahead_for_steer` | double | Lookahead time for steering calculation [s] | 0.0 - 0.2 |

### MAP-Specific Parameters

| Parameter | Type | Description | Range |
|-----------|------|-------------|-------|
| `diff_threshold` | double | Speed difference threshold for startup blending [m/s] | 0.0 - 20.0 |
| `deacc_gain` | double | Blending gain for startup deacceleration | 0.0 - 1.0 |

## Control Algorithms

### MAP Controller

**Steering Calculation Pipeline:**
1. Lookup steering angle from CSV table based on (velocity, lateral_acceleration)
2. Apply speed-based downscaling (linear from `start_scale_speed` to `end_scale_speed`)
3. Apply acceleration-based scaling (multiply by `acc_scaler_for_steer` or `dec_scaler_for_steer`)
4. Apply speed multiplier (1.0 to 1.25 based on current speed)
5. Apply rate limiting (0.4 rad/step, skipped on first calculation)
6. Clamp to ±0.45 rad

**L1 Lookahead Distance:**
```
L1_distance = q_l1 + speed_now * m_l1
L1_distance = clamp(L1_distance, max(t_clip_min, lateral_multiplier * lateral_error), t_clip_max)
```
where `lateral_multiplier = 2.0` if `lateral_error > 1.0m`, else `√2`.

**Speed Command:**
- Lookhead position speed with lateral error exponential reduction
- Startup blending when `|profile_speed - current_speed| >= diff_threshold`

### PP Controller

**Steering Calculation:**
Pure Pursuit geometric formula:
```
steering_angle = arctan((2 * wheelbase * sin(eta)) / L1_distance)
```
where `eta` is the angle between vehicle heading and L1 vector, `wheelbase = 0.33m`.

Then applies same scaling pipeline as MAP (steps 2-6 above).

**L1 Lookahead Distance:**
Same as MAP controller.

**Speed Command:**
Lookahead position speed with lateral error adjustment (no startup blending).

## Dynamic Reconfiguration

All controller parameters can be reconfigured at runtime using ROS2 parameter services:

```bash
ros2 param set /controller_manager t_clip_max 5.0
```

The controller will automatically update the internal algorithm parameters.

## Safety Features

- **Waypoint Timeout**: Stops vehicle if no waypoints received for 5 seconds
- **Graceful Shutdown**: Publishes emergency stop commands (speed=0, acceleration=-5.0) on shutdown signal (Ctrl+C, SIGTERM)
- **State Validation**: Checks for valid TF, waypoints, and sensor data before control

## Differences from C++ Implementation

The Python implementation preserves all control logic exactly as in the original C++ package:
- Same steering calculation pipeline and scaling stages
- Same L1 adaptive lookahead formula
- Same lateral error normalization and speed adjustment
- Same parameter ranges and dynamic reconfiguration

Key adaptations:
- Eigen → NumPy for linear algebra
- rclcpp → rclpy for ROS2 Python API
- yaml-cpp → PyYAML for configuration parsing
- std::optional → typing.Optional for nullable values
- Direct logger calls instead of callback wrappers

### Verified Against crazycontroller

The implementation has been cross-validated with the proven `crazycontroller` Python package:

**Critical fixes applied:**
1. ✅ **Lateral acceleration formula**: Added factor of 2 (`lat_acc = 2.0 * speed^2 * sin(eta) / L1_distance`)
2. ✅ **Steering sign preservation**: Lookup table now preserves sign of lateral acceleration
3. ✅ **NaN handling**: Steering interpolation truncates at first NaN value
4. ✅ **Acceleration buffer**: Unified to 10 samples (matching C++ and crazycontroller)

**Validated behaviors:**
- L1 distance calculation with lateral error bounds
- Speed-based and acceleration-based steering scaling
- Rate limiting (0.4 rad/step) with first-calculation skip
- Lateral error normalization and speed adjustment

## Troubleshooting

**No waypoints received:**
- Check that `/local_waypoints` topic is publishing
- Verify waypoint message format matches ae_hyu_msgs/WpntArray

**TF transform errors:**
- Ensure `map` → `base_link` transform is being broadcast
- Check TF tree with `ros2 run tf2_tools view_frames`

**Controller not responding:**
- Verify all required topics are publishing at expected rates
- Check parameter values are within valid ranges
- Enable debug logging: `ros2 run controller controller_manager --ros-args --log-level DEBUG`

## License

MIT License (same as original crazy_controller package)

## Credits

Ported from `crazy_controller` C++ ROS2 package.
Maintains exact control algorithm logic and parameter semantics.
