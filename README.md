# Controller Package

Python implementation of MAP and PP controllers for autonomous racing, ported from the `crazy_controller` C++ ROS2 package.

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
# Controller package — structure and how to run
```

## How to run

Build the workspace and source the environment, then use the launch file.

1) Build (from workspace root):

```bash
cd ~/AE-HYU_workspace/ae_hyu_bundle
colcon build --packages-select controller
source install/setup.bash
```

2) Launch examples:

```bash
# MAP (default):
ros2 launch controller controller_launch.py mode:=MAP env:=sim

# PP:
ros2 launch controller controller_launch.py mode:=PP env:=sim

# Quick: launch with default args (MAP / real):
ros2 launch controller controller_launch.py
```
