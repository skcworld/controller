# Important!

This package implements only the core control algorithm, with all auxiliary functions and performance-enhancing modules removed for research purposes. The controller is based on the implementation from
https://github.com/ForzaETH/race_stack.
I would like to thank the ETH Forza team for providing an excellent reference.

The accompanying paper is currently under review, so the codebase has not yet been fully cleaned or documented. Once the review process is complete, I plan to add detailed comments and improve readability. Thank you for your understanding.

## How to run

Build the workspace and source the environment, then use the launch file.

1) Build (from workspace root):

```bash
colcon build --packages-select controller
source install/setup.bash
```

2) Launch examples:

```bash
# MAP (default):
ros2 launch controller controller_launch.py mode:=MAP env:=sim

# PP:
ros2 launch controller controller_launch.py mode:=PP env:=sim

# AUG (Proposed):
ros2 launch controller controller_launch.py mode:=AUG env:=sim
