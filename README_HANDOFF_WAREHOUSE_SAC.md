# Bàn giao dự án AGV MuJoCo: MPPI + SAC speed scale

**Cập nhật:** 15/10/2026. **Workspace:** `/home/anhkhoa/mujoco_agv_ws`.

Tài liệu này dành cho người hoặc coding agent tiếp nhận dự án. Hãy đọc dữ liệu và mã hiện tại trước khi đề xuất thay đổi. Không tự chạy train hoặc benchmark dài trên máy của chủ dự án; chuẩn bị lệnh và để chủ dự án tự chạy. Làm từng vấn đề một, không tự tăng số trial.

## 1. Mục tiêu và phạm vi

Tối ưu đồng thời ba tiêu chí: (1) giữ vận tốc gần mức MPPI nền khi đường thực sự thoáng, (2) giảm va chạm khi vật cản động cắt ngang, và (3) không tăng timeout/deadlock. Chỉ so sánh hai map `mppi_warehouse_aisles` và `mppi_warehouse_mixed`; không đưa map hành lang tầng 2 vào các so sánh tiếp theo. Checkpoint 20k hiện tại **chưa đạt mục tiêu vận tốc**, dù tỷ lệ hoàn thành tăng.

## 2. Kiến trúc đang có

`SAC (1 action speed_scale)` → `/rl_velocity_limits` → `rl_velocity_supervisor.py` → `nav2_msgs/SpeedLimit` trên `/speed_limit` → Nav2 MPPI → `/cmd_vel_nav` → `/cmd_vel` → cầu nối bánh xe/MuJoCo.

- Action SAC nằm trong `[0.15, 1.0]`, cập nhật khoảng **1 Hz**. SAC không phát `cmd_vel` trực tiếp.
- Supervisor cập nhật **5 Hz**, lọc scale với slew rate `0.25 scale/s`, low-pass alpha `0.35`, watchdog `2.5 s`. Cấu hình SAC-only: `src/agv_mujoco_sim/config/rl_supervisor_sac_only.yaml`.
- Trần MPPI nền: `vx_max=0.70 m/s`, `wz_max=0.50 rad/s`. `SpeedLimit` dạng phần trăm hiện tác động lên **cả hai** trần vx và wz. Scale tối đa 1.0 không thể làm xe nhanh hơn MPPI thuần ở đoạn thoáng.
- Benchmark SAC-only tắt TTC override và front safety stop. Ý tưởng chỉ giới hạn vx, giữ nguyên wz **chưa triển khai**.
- Code chính: `src/agv_mujoco_sim/scripts/agv_rl_env.py`, `rl_velocity_supervisor.py`, `train_sac.py`, `benchmark_dynamic_obstacle.py`, `run_mppi_three_maps.py`.
- Ba tín hiệu vận tốc cần phân biệt: `/cmd_vel` là lệnh; `/odometry/filtered` là ước lượng từ odometry bánh + IMU qua EKF; `/ground_truth/odom` lấy vận tốc thân robot trực tiếp từ MuJoCo. Robot thật không có ground truth MuJoCo.

## 3. Checkpoint và dữ liệu đã có

- Checkpoint 20k: `runs/sac_only_conservative_20k_seed42/final_model.zip`. Hợp đồng model: observation 31D không TTC, action 1D, 1 Hz; xem `runs/sac_only_conservative_20k_seed42/training_summary.json`. Đây là lần tiếp tục từ checkpoint 10k, kết thúc ở 20k steps, không phải bằng chứng đã hội tụ.
- MPPI thuần: `benchmark_results/mppi_three_maps_20261011_235102/`.
- SAC-only 20k: `benchmark_results/sac_only_conservative_two_warehouse_seed42_retry/`.
- Mỗi arm có 100 scenario/map, seed 42, goal `(9,0)`, timeout 75 s. Danh sách scenario của hai arm trùng nhau; SHA-256 của JSON scenario chuẩn hóa là `3939bee34132e6da...` trên cả hai map/arm.
- Mỗi map có `benchmark.json`, `benchmark.samples.csv`, `run_metadata.json`, rosbag và log. CSV có vị trí, thời gian, SAC requested scale, supervisor applied scale, lệnh MPPI/lệnh cuối, odometry vx/wz. Rosbag còn có `/ground_truth/odom`.

| Arm/map | Thành công | Va chạm | Timeout | Vận tốc trước trigger (m/s) | Thời gian tới goal của lượt thành công (s) | Moving encounters |
|---|---:|---:|---:|---:|---:|---:|
| MPPI / aisles | 7/100 | 93 | 0 | 0.6996 | 28.74 | 100 |
| SAC 20k / aisles | 30/100 | 66 | 4 | 0.3328 | 46.37 | 6 |
| MPPI / mixed | 13/100 | 87 | 0 | 0.6997 | 21.18 | 99 |
| SAC 20k / mixed | 99/100 | 0 | 1 | 0.5123 | 36.21 | 40 |

Vận tốc trong bảng là `linear.x` từ `/odometry/filtered`, trung bình theo trial tại `0.5 <= x < trigger_x`, trước khi vật cản động hoạt động. “Trước trigger” **không đồng nghĩa toàn tuyến không có vật cản tĩnh**. Thời gian tới goal chỉ tính lượt thành công nên có selection bias. `moving_encounter` chỉ được tính khi robot cách tâm vật cản ≤ 2 m trong lúc vật cản còn chuyển động; mẫu số khác nhau nên không diễn giải thành “SAC tránh giao cắt 99%”.

## 4. Chẩn đoán đã làm trên dữ liệu 20k

Xem `reports/chan_doan_toc_do_sac_20k_hai_map_warehouse.md`. Ở đoạn trước trigger, trung bình đồng trọng số 100 trial:

| Map | Requested scale | Applied scale | MPPI cmd vx | Final cmd vx | Odom vx |
|---|---:|---:|---:|---:|---:|
| aisles | 0.453 | 0.473 | 0.331 | 0.332 | 0.333 m/s |
| mixed | 0.725 | 0.731 | 0.512 | 0.512 | 0.512 m/s |

Nút giảm tốc chủ yếu nằm ở **scale SAC yêu cầu**. Supervisor có trễ khi đổi scale nhưng applied, lệnh cuối và odometry bám nhau khá sát khi chạy ổn định. Ở aisles, request giảm sâu hơn khi LiDAR trước thấy vật cản gần; đây là tương quan, chưa chứng minh SAC phân loại vật cản tĩnh/động. Ngay cả khi `front_lidar_m > 4 m`, requested scale trung bình vẫn khoảng `0.72` ở cả hai map: policy 20k có xu hướng thận trọng cả nơi phía trước tương đối trống.

Reward hiện thưởng **vận tốc đo được**, đồng thời phạt request cao theo rủi ro trước/bên hông. `conservative_lateral_risk` từ scan hiện tại không phân biệt vật cản đứng yên và đang chuyển động. Training summary ghi tổng lateral-risk penalty lớn, nhưng CSV benchmark không lưu trực tiếp risk feature/reward từng quyết định; **chưa xác định được penalty nào là nguyên nhân nhân quả**. Không sửa reward bằng cách đơn giản thưởng `scale=1`.

Biểu đồ: `reports/actual_speed_mppi_vs_sac20k_two_warehouse.png` dùng `/odometry/filtered`; `reports/ground_truth_speed_mppi_vs_sac20k_two_warehouse.png` dùng `/ground_truth/odom` từ rosbag. Hai biểu đồ dùng cách gộp mẫu khác nhau, không so trực tiếp từng điểm giữa hai hình.

## 5. Việc cần làm tiếp, theo thứ tự

1. **Khóa hiện trạng và kiểm tra dữ liệu mới sau train/test.** Ghi đường dẫn checkpoint, SHA của model, contract 31D/1D, seed, config supervisor, map, scenario, số trial và phiên bản code. Kiểm tra model bằng `train_sac.py --inspect-only --sac-only --model <checkpoint>`. Không coi 20k steps là tối ưu mặc định.
2. **Đo cùng điều kiện trên đúng hai map.** So với MPPI thuần bằng cùng scenario/seed, cùng timeout và cùng định nghĩa collision/success. Báo cáo success/collision/timeout, vận tốc trước trigger, vận tốc khi front clearance lớn, thời gian tới goal, và moving encounters với mẫu số. Đối chiếu `/cmd_vel_nav`, `/cmd_vel`, `/odometry/filtered`; nếu cần tốc độ vật lý mô phỏng, dùng `/ground_truth/odom` từ rosbag.
3. **Tách nguyên nhân policy chậm.** Trước khi thay reward, bổ sung log theo mỗi quyết định SAC: `front_proactive_risk`, `conservative_lateral_risk`, tuổi scan, front margin, requested/applied scale và reward thành phần. Kiểm tra cả khi vật cản động chưa hoạt động và khi vật cản tĩnh nằm bên hông. CSV hiện tại không đủ để hồi tính đầy đủ các feature này.
4. **Chọn một thay đổi nhỏ rồi test lại.** Nếu risk/reward phạt quá rộng, chỉnh đúng feature/reward đó. Nếu lệnh quay bị kìm gây deadlock, thử nhánh chỉ giới hạn vx và giữ wz; đây là thay đổi kiến trúc riêng, không trộn với thay reward trong cùng phép so sánh. Với sửa supervisor/SpeedLimit, đánh giá lại **cùng checkpoint** trước khi train lại để tách tác động triển khai khỏi tác động học.
5. **Train tiếp chỉ khi đã có giả thuyết cụ thể.** Giữ seed/config/model/replay metadata. Sau train, test lại cùng kịch bản hai map. Không tự thêm trial hay map mới khi chủ dự án chưa yêu cầu.

Tiêu chí quyết định: không chỉ tăng success; cần giữ vận tốc gần MPPI nền khi thực sự thoáng, giảm va chạm ở các moving encounter, đồng thời không đổi va chạm lấy timeout/deadlock. Chưa đặt ngưỡng số học mới nếu nhóm chưa thống nhất.

## 6. Lệnh kiểm tra và benchmark cho người vận hành

Chạy trong terminal đã có ROS 2 Humble và workspace build phù hợp:

```bash
cd /home/anhkhoa/mujoco_agv_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Kiểm tra checkpoint, không khởi động mô phỏng:
ros2 run agv_mujoco_sim train_sac.py --inspect-only --sac-only \
  --model runs/sac_only_conservative_20k_seed42/final_model.zip

# Xem chính xác stack/lệnh benchmark dự kiến, chưa chạy benchmark:
ros2 run agv_mujoco_sim run_mppi_three_maps.py \
  --arm sac_only --model runs/sac_only_conservative_20k_seed42/final_model.zip \
  --map mppi_warehouse_aisles --map mppi_warehouse_mixed \
  --trials 100 --seed 42 --timeout-s 75 --dry-run
```

Khi chủ dự án muốn tự benchmark checkpoint mới, bỏ `--dry-run`, chỉ định `--output-dir` mới và không ghi đè dữ liệu cũ. Benchmark thực tế lâu; agent không tự chạy. Trước khi dùng checkpoint mới, kiểm tra model contract và cờ SAC-only tương thích. `run_mppi_three_maps.py` mặc định chạy **ba** map nếu thiếu cả hai tham số `--map`, nên luôn ghi rõ hai map warehouse.

## 7. Chia sẻ cho đồng nghiệp/agent khác

README này cung cấp bối cảnh, nhưng **không tự chuyển mã hoặc dữ liệu**. Repository có remote Git `origin` tại `github.com/aykayy1/mujoco-agv-ws`. Khi nhận nhánh bàn giao, hãy kiểm tra nó chứa mã nguồn, README, checkpoint 20k (`final_model.zip`), training summary, và `benchmark.json`/`benchmark.samples.csv` của hai map. Các thư mục `runs/` và `benchmark_results/` còn nhiều dữ liệu lịch sử không thuộc gói bàn giao. Rosbag lớn được giữ ở máy nguồn, không đưa vào Git; để kiểm tra bảng kết quả chỉ cần JSON/CSV, nhưng để tính lại ground truth từ đầu cần rosbag hoặc bản trích xuất riêng.

Prompt ngắn để đưa cho agent tiếp nhận:

> Đọc `README_HANDOFF_WAREHOUSE_SAC.md` và kiểm tra file/dữ liệu thực tế trước khi kết luận. Chỉ làm việc trên `mppi_warehouse_aisles` và `mppi_warehouse_mixed`. Đầu tiên đối chiếu checkpoint mới với baseline 20k và MPPI bằng cùng scenario; sau đó chẩn đoán requested → applied → MPPI cmd → final cmd → odometry/ground truth. Không tự chạy train hoặc benchmark dài; đưa lệnh để tôi chạy. Không thay reward và supervisor trong cùng một thí nghiệm.
