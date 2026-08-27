# MuJoCo Nav2 baseline scenarios

Ba môi trường nhỏ để kiểm tra Nav2 + MPPI độc lập với map tầng 2:

| Scenario | Mục tiêu | Start pose | Goal pose |
| --- | --- | --- | --- |
| `straight` | Chạy thẳng, đo thời gian và độ mượt | `(0, 0, 0)` | `(8.5, 0, 0)` |
| `l_corner` | Bám đường và quay góc 90 độ | `(0, 0, 0)` | `(5.0, 6.0, 1.5708)` |
| `unexpected_obstacle` | Né vật cản không có trong static map | `(0, 0, 0)` | `(8.5, 0, 0)` |

Trong preview: xanh dương là start, xanh lá là goal. Khối đỏ chỉ có trong MJCF,
không được vẽ vào PGM; vì vậy Nav2 chỉ có thể nhận nó qua `/scan` và
`obstacle_layer`.

## Cài bộ scenario vào workspace

Giả sử package nguồn nằm tại:

```bash
PKG=$HOME/mujoco_agv_ws/src/agv_mujoco_sim
TESTS=$HOME/mujoco_agv_ws/nav2_test_scenarios
```

Chép thư mục `nav2_test_scenarios` được cung cấp vào:

```text
~/mujoco_agv_ws/nav2_test_scenarios
```

Sau đó:

```bash
mkdir -p "$PKG/models" "$HOME/mujoco_agv_ws/maps/nav2_tests"

cp "$TESTS"/models/*.xml "$PKG/models/"
cp "$TESTS"/maps/* "$HOME/mujoco_agv_ws/maps/nav2_tests/"

cp -n "$PKG/models/agv_spawn.xml" \
  "$PKG/models/agv_spawn_floor2_backup.xml"
```

## Chọn scenario

Bridge hiện load cố định `models/agv_spawn.xml`. Trước khi chạy simulation,
chọn một wrapper làm model đang hoạt động.

Straight:

```bash
cp "$PKG/models/agv_spawn_test_straight.xml" \
  "$PKG/models/agv_spawn.xml"
```

L-corner:

```bash
cp "$PKG/models/agv_spawn_test_l_corner.xml" \
  "$PKG/models/agv_spawn.xml"
```

Unexpected obstacle:

```bash
cp "$PKG/models/agv_spawn_test_unexpected_obstacle.xml" \
  "$PKG/models/agv_spawn.xml"
```

Build sau khi cài file lần đầu hoặc sau khi đổi model nếu install tree không
phản ánh thay đổi nguồn:

```bash
cd ~/mujoco_agv_ws
colcon build --packages-select agv_mujoco_sim --symlink-install
source install/local_setup.bash
```

## Chạy simulation và Nav2

Terminal 1:

```bash
source /opt/ros/humble/setup.bash
source ~/mujoco_agv_ws/install/local_setup.bash
ros2 launch agv_mujoco_sim mujoco_sim.launch.py
```

Terminal 2, thay `<scenario>` bằng `straight`, `l_corner` hoặc
`unexpected_obstacle`:

```bash
source /opt/ros/humble/setup.bash
source ~/mujoco_agv_ws/install/local_setup.bash

ros2 launch nav2_bringup localization_launch.py \
  map:=$HOME/mujoco_agv_ws/maps/nav2_tests/nav2_test_<scenario>.yaml \
  params_file:=$HOME/mujoco_agv_ws/config/nav2_floor2_stock_humble.yaml \
  use_sim_time:=true \
  autostart:=true
```

Terminal 3:

```bash
source /opt/ros/humble/setup.bash
source ~/mujoco_agv_ws/install/local_setup.bash

ros2 launch nav2_bringup navigation_launch.py \
  params_file:=$HOME/mujoco_agv_ws/config/nav2_floor2_stock_humble.yaml \
  use_sim_time:=true \
  autostart:=true
```

Đặt initial pose tại `(0, 0, 0)` trong RViz trước khi gửi goal.

## Goal bằng command line

Straight và unexpected obstacle:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 8.5, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}" \
  --feedback
```

L-corner:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 5.0, y: 6.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.7071068, w: 0.7071068}}}}" \
  --feedback
```

## Khôi phục map tầng 2

```bash
cp "$PKG/models/agv_spawn_floor2_backup.xml" \
  "$PKG/models/agv_spawn.xml"

cd ~/mujoco_agv_ws
colcon build --packages-select agv_mujoco_sim --symlink-install
```

