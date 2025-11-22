# Important!

연구 목적을 위해 제어 성능을 높이기 위한 여러 함수, 코드들을 주석 처리하고, 순수한 제어 알고리즘만 구현된 패키지입니다.

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

# AUG :
ros2 launch controller controller_launch.py mode:=AUG env:=sim
