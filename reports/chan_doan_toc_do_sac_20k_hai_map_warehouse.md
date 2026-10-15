# Chẩn đoán tốc độ SAC 20k trên hai map warehouse

## Phạm vi và cách tính

Nguồn: `benchmark_results/sac_only_conservative_two_warehouse_seed42_retry/` (seed 42, 100 trial/map, checkpoint `runs/sac_only_conservative_20k_seed42/final_model.zip`). Chỉ đọc `benchmark.samples.csv` và `benchmark.json`; không chạy lại mô phỏng.

Đoạn *trước trigger*: `0.5 <= x < trigger_x` của chính trial đó và `obstacle_active=False`. Các trung bình ở bảng đầu là trung bình trên **100 trung bình trial**, đồng trọng số mỗi trial, để khớp cách báo cáo tốc độ đoạn thoáng trong benchmark. Các bảng theo vị trí/thời gian bên dưới là trung bình theo **mẫu 0.05 s**, nên trial đi chậm đóng góp nhiều mẫu hơn. `actual_vx` là tốc độ odometry có dấu, không phải `cmd_vel`. Sai khác rất nhỏ giữa tốc độ có dấu ở đây và phép lấy `max(0, vx)` của benchmark trên đoạn này.

## Bốn tầng tín hiệu trước trigger

| Map | Requested scale | Applied scale | MPPI cmd vx (m/s) | Final cmd vx (m/s) | Actual vx (m/s) |
|---|---:|---:|---:|---:|---:|
| aisles | 0.453 | 0.473 | 0.331 | 0.332 | 0.333 |
| mixed | 0.725 | 0.731 | 0.512 | 0.512 | 0.512 |

Trần MPPI nền là 0.70 m/s. Trung bình theo mẫu, `0.70 × applied_scale` gần đúng lệnh MPPI: aisles `0.275` so với `0.275` m/s, mixed `0.510` so với `0.510` m/s. Sai lệch tuyệt đối trung bình giữa `mppi_cmd_vx` và `final_cmd_vx` là 0.0032 m/s (aisles), 0.0015 m/s (mixed); giữa `final_cmd_vx` và `actual_vx` là 0.0091 và 0.0063 m/s. Như vậy, mất tốc độ chủ yếu xuất hiện ở yêu cầu của SAC. Không thấy nút nghẽn đáng kể từ việc chuyển lệnh MPPI sang lệnh cuối hoặc từ lệnh cuối sang odometry trong đoạn này.

Supervisor có độ trễ khi scale thay đổi, nhưng không giải thích chính cho vận tốc thấp kéo dài: applied cao hơn requested trung bình 0.020 (aisles) và 0.006 (mixed) nếu đồng trọng số trial. Ở aisles trong khoảng 3–6 s, trung bình theo mẫu là requested 0.484 và applied 0.591; lúc này bộ lọc còn giữ scale cao hơn yêu cầu mới. Trong khoảng 9–15 s, hai giá trị đều khoảng 0.20.

## Theo vị trí và thời gian

| Map và đoạn trước trigger | Số mẫu / trial | Requested | Applied | Actual vx (m/s) | Front lidar (m) |
|---|---:|---:|---:|---:|---:|
| aisles, x 0.5–1.5 m | 6,877 / 100 | 0.506 | 0.517 | 0.361 | 3.027 |
| aisles, x 1.5–2.5 m | 12,103 / 91 | 0.285 | 0.315 | 0.225 | 1.322 |
| mixed, x 0.5–1.5 m | 3,927 / 100 | 0.735 | 0.729 | 0.510 | 2.449 |
| mixed, x 1.5–2.5 m | 3,648 / 100 | 0.739 | 0.748 | 0.524 | 5.011 |

| Map và thời gian trước trigger | Số mẫu / trial | Requested | Applied | Actual vx (m/s) | Front lidar (m) |
|---|---:|---:|---:|---:|---:|
| aisles, 0–3 s | 2,032 / 100 | 0.702 | 0.684 | 0.473 | 4.299 |
| aisles, 3–6 s | 5,878 / 100 | 0.484 | 0.591 | 0.423 | 2.931 |
| aisles, 6–9 s | 4,116 / 87 | 0.313 | 0.307 | 0.217 | 1.439 |
| aisles, 9–15 s | 6,644 / 60 | 0.201 | 0.200 | 0.141 | 0.766 |
| mixed, 0–3 s | 2,319 / 100 | 0.735 | 0.734 | 0.515 | 2.319 |
| mixed, 3–6 s | 5,624 / 100 | 0.734 | 0.737 | 0.516 | 4.293 |
| mixed, 6–9 s | 907 / 64 | 0.615 | 0.660 | 0.467 | 4.221 |

Ở aisles, khoảng 1.5–2.5 m có nhiều mẫu gần vật cản tĩnh. Do đó gọi toàn bộ đoạn trước trigger là “đường thoáng” chỉ có nghĩa **vật cản động chưa kích hoạt**, không có nghĩa trước robot trống. Tuy nhiên, ngay cả các mẫu trước trigger có `front_lidar_m > 4` m, requested/applied/actual vẫn lần lượt là 0.715/0.698/0.487 m/s ở aisles (2,666 mẫu, 75 trial) và 0.721/0.734/0.515 m/s ở mixed (4,849 mẫu, 100 trial). Đây là bằng chứng rằng policy 20k chưa duy trì trần nền 0.70 m/s cả khi khoảng cách phía trước lớn; chưa thể quy toàn bộ nguyên nhân cho vật cản tĩnh ở hành lang.

## Diễn giải và bước kế tiếp

1. Ưu tiên chẩn đoán **SAC yêu cầu scale thấp**. Với các mẫu trước trigger, 93.6% ở aisles và 86.0% ở mixed có requested scale dưới 0.8. Supervisor làm mượt gây trễ chuyển tiếp nhưng không phải nguyên nhân chủ đạo của tốc độ thấp ổn định.
2. Tương quan giữa khoảng trống trước robot và scale ở aisles gợi ý policy phản ứng với hình học vật cản tĩnh. Reward hiện tại phạt request cao theo rủi ro trước và bên hông; `conservative_lateral_risk` dùng scan hiện thời, không nhận dạng vật thể đang chuyển động. Song CSV chỉ lưu `front_lidar_m` và `min_lidar_m`, **không lưu front/lateral risk, front scan tuổi, hoặc reward từng bước**. Chưa thể kết luận chính xác penalty nào tạo ra request thấp; cũng chưa thể tính lại risk đầy đủ từ CSV này.
3. Nếu muốn tách nguyên nhân ở mức feature/reward, bước nhỏ tiếp theo là ghi trực tiếp `front_proactive_risk`, `conservative_lateral_risk` và các reward thành phần lúc policy quyết định, sau đó chạy kiểm tra ngắn với cùng checkpoint và scenario. Chưa sửa reward hoặc train lại dựa riêng trên bảng tương quan này.

Giới hạn: các thống kê theo mẫu có trọng số thời gian và không độc lập giữa các hàng; bảng không ước lượng độ tin cậy thống kê hay hiệu quả nhân quả. `front_lidar_m > 4` chỉ xác nhận một chỉ số khoảng trống phía trước lớn, không xác nhận toàn bộ vùng quét/đường đi không có rủi ro.
