# Đo và ghi dữ liệu robot thật cho sim2real

Nguồn đối chiếu: `/home/anhkhoa/articubot_one_23_4/articubot_one` (bản robot thật), ngày 06-10-2026. Tài liệu này **không** khởi động SAC, không phát lệnh chạy bánh xe. Người vận hành tự xác nhận phần cứng, khu vực thử và nút dừng khẩn cấp.

## 0. Khác biệt phải nhớ trước khi đo

| Thành phần | Trong `articubot_one` | Cần xác nhận trên robot đang chạy |
|---|---|---|
| LiDAR thô | `/scan`, RPLidar A2M12 trong `sensors.launch.py` | Model thật, `frame_id`, số tia, tần số, góc và QoS |
| LiDAR sau lọc | `/scan_filtered`, lọc góc ±1,04719 rad (±60°) rồi lọc hộp thân xe | Có mất vật cản thật ở gần/sát mép vùng nhìn không |
| Vận tốc bánh đo được | `/wheel_vel` (`TwistWithCovarianceStamped`), `/wheel_vel_raw` (RPM dạng chuỗi) | Có dữ liệu thật và đúng dấu khi chạy tới/quay |
| Odom EKF | `/odometry/filtered`, TF `odom → base_foot_link` | EKF thực sự hoạt động, `use_sim_time=false` |
| IMU | `/imu`, `frame_id=imu` trong `khoa.py` | Tần số, chiều yaw, dữ liệu khi đứng yên |
| Lệnh xuống motor | `wheel_vel_node.py` subscribe `/cmd_vel` (`Twist`) | Nguồn phát lệnh thực tế, mux, E-stop, timeout |
| Vị trí LiDAR URDF | `base_link → lidar`: (0,33152; 0; 0,11235) m, yaw ≈ 180° | Đúng với lắp đặt thật và chiều quét phía trước |

**Cảnh báo cấu hình:** `sensors.launch.py` mặc định motor `/dev/ttyACM0` ở 57600 baud, nhưng `wheel_vel_node.py` mặc định `/dev/ttyUSB1` ở 115200 baud; phần chú thích trong code lại mô tả một sơ đồ cổng khác. `lidar.launch.py` dùng frame `Lidar`, `rplidar.launch.py` dùng `laser_frame`, còn `sensors.launch.py` dùng `lidar`. Không dựa vào tên cổng/frame mặc định; đo trực tiếp trên robot. `joystick.launch.py` xuất `/cmd_vel_joy`, trong khi motor đọc `/cmd_vel`; không mặc nhiên coi joystick đã nối vào motor. `twist_mux.yaml` đang khai báo đầu vào `cmd_vel` và một số launch remap đầu ra về `/cmd_vel`: phải kiểm tra graph để tránh vòng lặp/tranh chấp trước mọi phép thử chuyển động.

## 1. Chuẩn bị và kiểm tra tĩnh — chưa cho robot chạy

- [ ] Đặt robot trên nền phẳng, khu vực trống, có người giám sát và E-stop hoạt động. Không bật SAC/Nav2 tự chạy.
- [ ] Ghi ngày, cấu hình pin/tải trọng/mặt sàn, model LiDAR, chiều cao và vị trí tâm LiDAR đo bằng thước.
- [ ] Mở terminal đã `source` **đúng workspace của robot thật**. Không `source` workspace MuJoCo vào robot nếu đó không phải bản triển khai trên robot.
- [ ] Kiểm tra cổng vật lý, tuyệt đối chưa chạy launch motor nếu chưa biết cổng nào thuộc motor/IMU/LiDAR.

```bash
echo "$ROS_DISTRO"
ros2 pkg prefix articubot_one
ls -l /dev/serial/by-id/
ls -l /dev/ttyUSB* /dev/ttyACM*
ros2 node list
ros2 topic list -t
```

Nếu robot **chưa chạy các node cảm biến**, chỉ sau khi đã xác nhận 3 cổng và baud trên robot, có thể khởi động `sensors.launch.py` với giá trị đã kiểm chứng. Lệnh này cũng khởi động `wheel_vel_node` và ghi lệnh 0 RPM khi chưa có `/cmd_vel`; vẫn phải giữ E-stop sẵn sàng. Thay ba đường dẫn và baud bên dưới bằng giá trị thật trước khi chạy; không chạy nguyên văn dấu `<...>`:

```bash
ros2 launch articubot_one sensors.launch.py \
  lidar_port:=<CONG_LIDAR_DA_XAC_NHAN> \
  imu_port:=<CONG_IMU_DA_XAC_NHAN> \
  wheel_port:=<CONG_MOTOR_DA_XAC_NHAN> \
  wheel_baudrate:=<BAUD_MOTOR_DA_XAC_NHAN>
```

Nếu các node đã chạy, **không chạy launch lần hai**. Với bản robot thật, `mapping.launch.py` có EKF và robot_state_publisher nhưng cũng bật SLAM/mux; chỉ dùng launch đang vận hành ổn định. Không dùng `voxel_nav2.launch.py` chỉ để thu dữ liệu vì launch đó bật thêm nhiều thành phần điều hướng.

## 2. Kiểm tra ROS/TF/topic khi đứng yên

Các lệnh `hz` và `tf2_echo` chạy liên tục; nhấn `Ctrl+C` sau 20–30 giây. Nếu topic/node chưa tồn tại, ghi “không có”, không tự đổi tên theo mô phỏng.

```bash
ros2 topic info /scan -v
ros2 topic info /scan_filtered -v
ros2 topic info /wheel_vel -v
ros2 topic info /cmd_vel -v
ros2 topic echo /scan --once --no-arr
ros2 topic echo /scan_filtered --once --no-arr
ros2 topic echo /wheel_vel --once
ros2 topic echo /imu --once --no-arr
ros2 topic hz /scan --window 100
ros2 topic hz /scan_filtered --window 100
ros2 topic hz /wheel_vel --window 100
ros2 topic hz /imu --window 100
ros2 topic hz /odometry/filtered --window 100
ros2 run tf2_ros tf2_echo base_link lidar
ros2 run tf2_ros tf2_echo odom base_foot_link
ros2 param get /ekf_filter_node use_sim_time
```

Ghi kết quả: `/scan` ___ Hz; `/scan_filtered` ___ Hz; số tia scan thô ___; `angle_min` ___; `angle_max` ___; `angle_increment` ___; `range_min/max` ___/___ m; frame scan ___; frame sau lọc ___; `/wheel_vel` ___ Hz; `/imu` ___ Hz; odom ___ Hz; TF LiDAR đúng/sai ___; EKF `use_sim_time` ___. Với robot thật phải là `false`.

Nếu `odom → base_foot_link` chưa tồn tại, hoặc `base_link → lidar` không đúng frame/góc lắp, **dừng tại bước tĩnh**. TTC dựa trên lịch sử scan và góc yaw; thiếu/sai hai dữ liệu này sẽ làm so sánh sim–real vô nghĩa.

## 3. Ghi rosbag tĩnh

Trước khi ghi, xem danh sách topic ở mục 1. **Bỏ các topic không tồn tại** khỏi lệnh (ví dụ `/odometry/filtered` nếu chưa chạy EKF); ghi rõ topic nào thiếu. Ghi trong thư mục mới, đủ dung lượng. Dùng `Ctrl+C` để đóng bag, không tắt nguồn giữa chừng.

```bash
mkdir -p "$PWD/sim2real_bags"
ros2 bag record -o "$PWD/sim2real_bags/static_01" \
  /scan /scan_filtered /wheel_vel /wheel_vel_raw /imu \
  /odometry/filtered /cmd_vel /tf /tf_static \
  /power/status /power/voltage_left /power/voltage_right \
  /power/temp_left /power/temp_right
```

Trong 2–3 phút ghi bag, lần lượt làm và ghi thời điểm theo đồng hồ:

1. 30 giây không có vật cản phía trước: đo nhiễu tĩnh, thân robot lọt scan, tia `inf`/`nan`, scan bị rớt.
2. Đặt vật phẳng tĩnh tại 0°, khoảng cách 0,5 / 1,0 / 2,0 m **tính từ tâm LiDAR**; không che/tác động trực tiếp vào LiDAR.
3. Lặp ở hai bên ±30° và gần ±60°. Với mỗi vị trí, ghi khoảng cách thước và liệu `/scan_filtered` còn thấy vật hay không.
4. Thử vật thấp/cao, bề mặt tối/sáng/phản quang nếu chúng xuất hiện trong môi trường sử dụng. Ghi loại vật và kích thước.

Mẫu ghi tay: `thời gian | góc | khoảng cách thước | loại/kích thước vật | thấy ở /scan? | thấy ở /scan_filtered? | nhận xét`.

```bash
ros2 bag info "$PWD/sim2real_bags/static_01"
```

## 4. Đo chuyển động và quãng dừng — chỉ sau khi mục 1–3 đạt

**Không dùng lệnh `ros2 topic pub /cmd_vel` để thử nhanh.** Trước hết xác nhận E-stop vật lý, nguồn `/cmd_vel` chỉ có đường điều khiển dự kiến, robot có thể dừng bằng bộ điều khiển đã kiểm nghiệm và người vận hành đứng ngoài đường chạy. Nếu `twist_mux` đang tự lặp hoặc `/cmd_vel_joy` chưa được ghép vào `/cmd_vel`, sửa/kiểm chứng đường lệnh riêng trước khi tiếp tục.

Mỗi phép đo dùng lại lệnh `ros2 bag record` ở mục 3, đổi `-o` thành tên bag **chưa tồn tại**:

- [ ] `rotate_01`: quay tại chỗ rất chậm bằng bộ điều khiển đã kiểm chứng; vật cản **đứng yên** quanh robot. Ghi chiều yaw IMU/EKF, thời gian đáp ứng, hiện tượng TTC giả khi quay.
- [ ] `straight_01`: chạy thẳng chậm, bắt đầu khoảng 0,1 m/s ở hành lang trống; đo quãng đường bằng thước và odom. Lặp tiến/lùi chỉ khi lùi đã được kiểm chứng an toàn.
- [ ] `stop_01`: từ tốc độ thấp đã đo, ra lệnh dừng qua bộ điều khiển đã kiểm chứng; đo từ vị trí ra lệnh đến điểm dừng hoàn toàn và thời gian dừng. Lặp ≥5 lần, ghi tải trọng/mặt sàn/điện áp. **Chưa thử tốc độ 0,70 m/s** của cấu hình mô phỏng.
- [ ] `turn_01`: chạy cong chậm; đo bán kính/quãng đường và so với odom để hiệu chỉnh bán kính bánh, khoảng cách bánh, độ trượt.
- [ ] `crossing_static_robot_01`: robot đứng yên, dùng vật/cụm vật không phải người đi cắt qua vùng quét phía trước; đo độ trễ phát hiện và TTC. Không cho vật chuyển động cắt đường chạy của robot ở giai đoạn này.

Ghi bảng riêng: `bag | tốc độ lệnh | tốc độ /wheel_vel | tốc độ EKF | quãng đường thước | quãng dừng | thời gian dừng | tải trọng | mặt sàn | điện áp | nhận xét`.

## 5. Kiểm tra mất dữ liệu và điều kiện chặn triển khai SAC

- [ ] Khi robot **đứng yên**, thử tắt node LiDAR có kiểm soát (không rút dây lúc robot đang chạy), xác nhận `/scan`/`/scan_filtered` mất và hệ thống điều khiển xử lý an toàn. Sau thử nghiệm phải khởi động lại đúng quy trình.
- [ ] Đo khoảng trống dài nhất giữa các scan/odom/IMU, độ lệch timestamp giữa các nguồn và số thông điệp bị mất từ bag; nếu các node chạy trên nhiều máy, ghi cấu hình đồng bộ đồng hồ (`date -Ins`, `timedatectl status`, và `chronyc tracking` nếu có chrony).
- [ ] Kiểm tra `/power/status`, điện áp/tốc độ thực khi pin yếu và khi tải thay đổi.
- [ ] Xác nhận scan-filter không xóa vật cản thật gần thân và vẫn phát hiện được vật tại biên ±60°.
- [ ] Có cơ chế dừng khi scan quá hạn, odom/TF lỗi, không còn lệnh, lỗi truyền thông motor; phải thử trên robot đứng yên trước. Timeout `/cmd_vel` 0,5 s trong `wheel_vel_node.py` **không thay thế** watchdog mất LiDAR.
- [ ] E-stop phần cứng và khả năng dừng độc lập với ROS đã được kiểm tra bởi người phụ trách robot.

**Chưa đưa SAC lên robot** cho tới khi hoàn thành các bước trên và xử lý các sai khác đo được. Trong code mô phỏng hiện tại, supervisor dùng 24 sector TTC, nhìn lùi 0,35 s và lịch sử tối đa 0,85 s; toàn bộ giả định về mật độ/tần số scan phải được kiểm tra bằng bag robot thật. Watchdog mất scan hoàn toàn là điều kiện an toàn cần bổ sung/xác nhận trước khi cho robot tự chạy. Tham khảo cơ chế `source_timeout` của [Nav2 Collision Monitor](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/collision_monitor/configuring_collision_monitor_node/); đây vẫn không thay thế E-stop phần cứng.

## 6. Gửi lại để hiệu chỉnh sim2real

Gửi các thư mục bag `static_01`, `rotate_01`, `straight_01`, `stop_01`, `turn_01`, `crossing_static_robot_01` đã thực hiện, bản ghi đo tay, đầu ra `ros2 topic list -t`, `ros2 topic info /scan -v`, `/scan_filtered -v`, `/cmd_vel -v`, và model/serial LiDAR cùng cấu hình cổng thực tế. Nếu không chuyển được bag, gửi `ros2 bag info <tên_bag>` và nêu giới hạn dung lượng; bag đầy đủ vẫn cần để đo độ trễ, dropout và phân bố nhiễu.
