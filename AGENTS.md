# Hướng dẫn agent tiếp nhận

Đọc `README_HANDOFF_WAREHOUSE_SAC.md` trước khi thay đổi dự án AGV MuJoCo/MPPI/SAC. Kiểm tra mã nguồn và dữ liệu thực tế, vì README là ảnh chụp trạng thái tại ngày ghi trong file.

- Phạm vi so sánh hiện tại: `mppi_warehouse_aisles` và `mppi_warehouse_mixed`.
- Mục tiêu đồng thời: giữ tốc độ gần MPPI nền khi đường thoáng, giảm va chạm lúc vật cản động cắt ngang, không tăng timeout/deadlock.
- Phân biệt lệnh `/cmd_vel`, ước lượng `/odometry/filtered`, và ground truth mô phỏng `/ground_truth/odom`.
- Không tự chạy train hoặc benchmark dài trên máy của chủ dự án. Chuẩn bị lệnh, kiểm tra ngắn khi cần, rồi để chủ dự án chạy tác vụ dài.
- Mỗi lần chỉ thử một thay đổi có thể quy nguyên nhân. Giữ seed/scenario/số trial nhất quán và lưu nguồn checkpoint, config, kết quả.
- Không đưa rosbag, replay buffer hoặc dữ liệu lịch sử nhiều GB lên Git nếu không có yêu cầu cụ thể.

Yêu cầu trực tiếp của người dùng luôn được ưu tiên hơn hướng dẫn trong file này.
