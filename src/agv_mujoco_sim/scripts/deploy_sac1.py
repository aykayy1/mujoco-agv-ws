#!/usr/bin/env python3
"""Chạy thực (inference) model SAC để tinh chỉnh 4 tham số MPPI.

Phiên bản này viết LẠI cho đúng API của agv_rl_env.AgvRlEnv hiện tại
(bản cũ gọi Nav2Env với hàng chục method không tồn tại -> crash ngay
từ import).

Ánh xạ API:
    _get_observation()          -> _observation()  (trả (obs, info))
    _apply_action_to_mppi()     -> _apply_action()
    _send_nav_goal(x, y)        -> ros.send_navigation_goal(x, y, yaw)
    _cancel_nav_goal_and_wait() -> ros.cancel_navigation()
    robot_x/robot_y/robot_yaw   -> ros.get_map_pose()
    min_obstacle_distance       -> info["min_lidar"]
    _nav_result_status          -> info["nav_status"]

LƯU Ý: KHÔNG gọi rclpy.spin_once() ở đây. AgvRlEnv đã chạy
MultiThreadedExecutor trong thread nền; spin thêm sẽ xung đột.

CÁCH DÙNG:
    python3 deploy_policy2.py
"""

import csv
import math
import os
import time

from action_msgs.msg import GoalStatus
from stable_baselines3 import SAC

from agv_rl_env import AgvRlEnv

# ==========================================
# CẤU HÌNH KỊCH BẢN
# ==========================================
MODEL_PATH = "/home/ubunturic/mujoco-agv-ws-main/src/agv_mujoco_sim/scripts/phase2_finetune_runs/run_20260827_110730/latest_model.zip"
LOG_CSV_PATH = "vmax_deploy_log.csv"
LOG_PLOT_PATH = "vmax_deploy_plot.png"
ERROR_CSV_PATH = "vmax_deploy_position_error.csv"

# 4 vị trí cố định (frame 'map'), dạng (x, y, yaw_mục_tiêu_rad).
# Phải giống hệt danh sách trong run_nav2_only.py để so sánh công bằng.
FIXED_GOALS = [
    (9.0, 0.0, 0.0),
    (-9.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (-3.0, 1.5, 0.0),
]

MAX_STEPS_PER_GOAL = 300
GOAL_ABORT_MAX_RETRY = 2


def _normalize_angle(angle_rad):
    """Đưa góc về [-pi, pi] để tính sai số ngắn nhất."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _wrap_deg_0_360(angle_deg):
    """Quy góc (độ) về [0, 360) — không còn số âm."""
    return angle_deg % 360.0


def _record_goal_error(error_rows, goal_idx, goal, env, ket_qua, goal_start_time):
    """Ghi sai số vị trí + góc cuối so với goal, kèm thời gian hoàn thành."""
    gx, gy, target_yaw = goal
    try:
        final_x, final_y, final_yaw = env.ros.get_map_pose()
    except Exception as exc:
        print(f"[deploy] Không đọc được pose cuối: {exc!r}")
        final_x = final_y = final_yaw = float("nan")

    elapsed_sec = time.time() - goal_start_time
    err_x = final_x - gx
    err_y = final_y - gy
    err_2d = math.hypot(err_x, err_y)
    err_yaw_signed = _normalize_angle(final_yaw - target_yaw)

    final_yaw_deg = _wrap_deg_0_360(math.degrees(final_yaw))
    err_yaw_deg = _wrap_deg_0_360(math.degrees(err_yaw_signed))
    err_yaw_abs_deg = abs(math.degrees(err_yaw_signed))

    error_rows.append((
        goal_idx, gx, gy, final_x, final_y, final_yaw_deg,
        err_x, err_y, err_yaw_deg, err_yaw_abs_deg,
        err_2d, ket_qua, round(elapsed_sec, 2),
    ))

    print(
        f"[deploy] >> Sai số goal #{goal_idx} ({ket_qua}, {elapsed_sec:.1f}s): "
        f"dx={err_x:+.3f}m dy={err_y:+.3f}m | 2D={err_2d:.3f}m | "
        f"yaw cuối={final_yaw_deg:.1f}° sai số yaw={err_yaw_deg:.1f}° "
        f"(ngắn nhất: {err_yaw_abs_deg:.1f}°)"
    )


def build_env():
    """AgvRlEnv chỉ nhận 4 đối số; mọi bound action đã hardcode trong class."""
    return AgvRlEnv(
        goals=FIXED_GOALS,
        goal_sampling="fixed",
        max_episode_steps=MAX_STEPS_PER_GOAL,
    )


def run_scenario():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy model tại '{MODEL_PATH}'. Sửa MODEL_PATH."
        )

    env = build_env()          # __init__ đã gọi ros.wait_until_ready()
    model = SAC.load(MODEL_PATH, env=env)
    print(f"[deploy] Đã load model, num_timesteps: {model.num_timesteps:,}")

    log_rows = []
    error_rows = []
    t_start = time.time()

    try:
        # Costmap đến qua TRANSIENT_LOCAL, thường có ngay; chờ tối đa 5s.
        deadline = time.time() + 5.0
        while env.ros.costmap is None and time.time() < deadline:
            time.sleep(0.1)

        for gx, gy, _ in FIXED_GOALS:
            if env.ros.costmap is not None and not env.ros.is_position_free(gx, gy):
                print(f"[deploy] CẢNH BÁO: goal ({gx:.2f}, {gy:.2f}) KHÔNG free trên costmap!")

        print(f"[deploy] Chạy tuần tự {len(FIXED_GOALS)} vị trí cố định:")
        for i, (gx, gy, _) in enumerate(FIXED_GOALS):
            print(f"    #{i}: ({gx:.2f}, {gy:.2f})")

        for goal_idx, goal in enumerate(FIXED_GOALS):
            gx, gy, gyaw = goal
            print(f"\n[deploy] === Goal #{goal_idx}: ({gx:.2f}, {gy:.2f}) ===")
            goal_start_time = time.time()

            env.goal = goal
            env.ros.cancel_navigation()
            env.ros.reset_episode_flags()      # xoá nav_status + cờ va chạm cũ
            env.ros.send_navigation_goal(gx, gy, gyaw)

            abort_retry_count = 0
            step_count = 0

            while True:
                step_count += 1

                obs, _ = env._observation()
                action, _ = model.predict(obs, deterministic=True)

                # _apply_action đã clip theo ACTION_LOW/HIGH rồi gửi SetParameters
                v_final, w_path, w_goal, w_goal_angle = env._apply_action(action)

                # Chờ đúng chu kỳ như lúc train. KHÔNG spin_once ở đây.
                deadline = time.monotonic() + env.STEP_WAIT_DURATION
                while time.monotonic() < deadline:
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

                _, info = env._observation()
                dist_to_goal = float(info["distance_to_goal"])
                d_obs = float(info["min_lidar"])
                nav_status = info["nav_status"]

                # Vận tốc THỰC của robot (đọc từ /odometry/filtered qua
                # AgvRosInterface._odom_callback) — khác với v_final, vốn
                # chỉ là setpoint vx_max gửi cho MPPI. MPPI/Nav2 không nhất
                # thiết đạt tới vx_max ngay (do gia tốc giới hạn, vật cản,
                # đang bo cua...), nên 2 giá trị này thường lệch nhau.
                state = env.ros.current_state()
                v_actual = float(state["linear_velocity"])
                w_actual = float(state["angular_velocity"])

                log_rows.append((
                    time.time() - t_start, goal_idx, step_count,
                    float(v_final), float(w_path), float(w_goal),
                    float(w_goal_angle), dist_to_goal, d_obs,
                    v_actual, w_actual,
                ))

                print(
                    f"[deploy] goal#{goal_idx} step={step_count:4d} "
                    f"vmax_dat={v_final:.2f} v_thuc={v_actual:.2f} w_thuc={w_actual:.2f} "
                    f"w_path={w_path:.2f} w_goal={w_goal:.2f} "
                    f"w_goal_angle={w_goal_angle:.2f}  dist={dist_to_goal:.2f}m "
                    f"d_obs={d_obs:.2f}m nav_status={nav_status}"
                )

                collided = (
                    bool(info["collision_state"])
                    or d_obs < env.COLLISION_DISTANCE
                    or int(info["contact_count"]) > 0
                )
                if collided:
                    print("[deploy] !!! VA CHẠM — dừng kịch bản.")
                    env.ros.cancel_navigation()
                    _record_goal_error(error_rows, goal_idx, goal, env,
                                       "COLLISION", goal_start_time)
                    return log_rows, error_rows

                if nav_status == GoalStatus.STATUS_SUCCEEDED:
                    print(f"[deploy] Goal #{goal_idx} THÀNH CÔNG sau {step_count} step.")
                    _record_goal_error(error_rows, goal_idx, goal, env,
                                       "SUCCEEDED", goal_start_time)
                    break

                if nav_status == GoalStatus.STATUS_ABORTED:
                    abort_retry_count += 1
                    print(f"[deploy] Goal #{goal_idx} ABORTED (lần {abort_retry_count}).")
                    if abort_retry_count > GOAL_ABORT_MAX_RETRY:
                        print(f"[deploy] Bỏ qua goal #{goal_idx}.")
                        _record_goal_error(error_rows, goal_idx, goal, env,
                                           "ABORTED", goal_start_time)
                        break
                    env.ros.send_navigation_goal(gx, gy, gyaw)

                if step_count >= MAX_STEPS_PER_GOAL:
                    print(f"[deploy] Goal #{goal_idx} quá {MAX_STEPS_PER_GOAL} step — bỏ qua.")
                    env.ros.cancel_navigation()
                    _record_goal_error(error_rows, goal_idx, goal, env,
                                       "TIMEOUT", goal_start_time)
                    break

    finally:
        env.close()

    return log_rows, error_rows


def save_log(log_rows):
    with open(LOG_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_wall_sec", "goal_idx", "step",
            "vmax_dat", "w_path", "w_goal", "w_goal_angle",
            "dist_to_goal_m", "d_obs_m",
            "v_thuc_mps", "w_thuc_radps",
        ])
        writer.writerows(log_rows)
    print(f"\n[deploy] Đã lưu log CSV: {LOG_CSV_PATH}")


def save_error_log(error_rows):
    with open(ERROR_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "goal_idx", "goal_x", "goal_y",
            "final_x", "final_y", "final_yaw_deg",
            "error_x_m", "error_y_m",
            "error_yaw_deg_0_360", "error_yaw_deg_shortest",
            "error_2d_m", "ket_qua", "thoi_gian_giay",
        ])
        writer.writerows(error_rows)

    if error_rows:
        mean_err_2d = sum(r[10] for r in error_rows) / len(error_rows)
        mean_err_yaw = sum(r[9] for r in error_rows) / len(error_rows)
        mean_time = sum(r[12] for r in error_rows) / len(error_rows)
        print(
            f"[deploy] Đã lưu sai số: {ERROR_CSV_PATH} "
            f"(2D TB: {mean_err_2d:.3f}m, yaw TB ngắn nhất: {mean_err_yaw:.1f}°, "
            f"thời gian TB: {mean_time:.1f}s / {len(error_rows)} goal)"
        )
    else:
        print(f"[deploy] Đã lưu file sai số (rỗng): {ERROR_CSV_PATH}")


def plot_log(log_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[deploy] Chưa cài matplotlib -> bỏ qua vẽ hình.")
        return

    if not log_rows:
        print("[deploy] Không có dữ liệu để vẽ.")
        return

    t = [r[0] for r in log_rows]
    fig, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(t, [r[3] for r in log_rows], label="vmax (đặt)", color="tab:blue", linestyle="--")
    axes[0].plot(t, [r[9] for r in log_rows], label="v thực", color="tab:cyan")
    axes[0].set_ylabel("Vận tốc dài (m/s)")
    axes[0].set_title("4 tham số MPPI + vận tốc thực theo thời gian (SAC policy)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, [r[10] for r in log_rows], color="tab:cyan")
    axes[1].set_ylabel("Vận tốc góc thực (rad/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, [r[4] for r in log_rows], label="w_path", color="tab:purple")
    axes[2].plot(t, [r[5] for r in log_rows], label="w_goal", color="tab:orange")
    axes[2].plot(t, [r[6] for r in log_rows], label="w_goal_angle", color="tab:brown")
    axes[2].set_ylabel("Critic weight")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, [r[7] for r in log_rows], color="tab:green")
    axes[3].set_ylabel("Khoảng cách tới goal (m)")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t, [r[8] for r in log_rows], color="tab:red")
    axes[4].set_ylabel("Vật cản gần nhất (m)")
    axes[4].set_xlabel("Thời gian thực (giây)")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(LOG_PLOT_PATH, dpi=150)
    print(f"[deploy] Đã lưu hình: {LOG_PLOT_PATH}")


if __name__ == "__main__":
    rows, err_rows = run_scenario()
    save_log(rows)
    save_error_log(err_rows)
    plot_log(rows)
    print("\n[deploy] HOÀN TẤT kịch bản.")