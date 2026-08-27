#!/usr/bin/env python3
"""
Script TIẾP TỤC huấn luyện từ lần chạy trước bị ngắt (Ctrl+C hoặc crash) 
cho môi trường MuJoCo AgvRlEnv.

CÁCH DÙNG:
    # Mặc định tiếp tục run_dir mới nhất hoặc chỉ định cụ thể:
    python3 resume_train.py --run-dir phase2_finetune_runs/run_2026... --total-steps 100000
"""

import argparse
import os
import time
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from agv_rl_env import AgvRlEnv

# Giữ nguyên giới hạn thread để tối ưu CPU
torch.set_num_threads(2)

def parse_args():
    parser = argparse.ArgumentParser(description="Resume training cho AgvRlEnv")
    parser.add_argument(
        "--run-dir", 
        type=Path, 
        required=True,
        help="Đường dẫn tới thư mục run_dir của lần train trước (vd: phase2_finetune_runs/run_...)"
    )
    parser.add_argument(
        "--total-steps", 
        type=int, 
        default=100000,
        help="MỐC TỔNG cuối cùng muốn đạt được (ORIGINAL_TOTAL_STEPS)"
    )
    return parser.parse_args()


class ProgressTrackerCallback(BaseCallback):
    """Callback in tiến độ % + reward trung bình theo thời gian thực."""
    def __init__(self, total_timesteps_target, print_freq=100, verbose=0):
        super(ProgressTrackerCallback, self).__init__(verbose)
        self.total_timesteps_target = total_timesteps_target
        self.print_freq = print_freq
        self._start_time = None
        self._start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._start_num_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            percent = (self.num_timesteps / self.total_timesteps_target) * 100
            elapsed = time.time() - self._start_time
            remaining_steps_this_session = self.total_timesteps_target - self.num_timesteps
            steps_done_this_session = max(self.num_timesteps - self._start_num_timesteps, 1)
            steps_per_sec = steps_done_this_session / elapsed if elapsed > 0 else 0.0

            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in ep_info_buffer]
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                reward_str = f"{mean_reward:.2f}"
            else:
                reward_str = "chưa có episode nào hoàn thành"

            eta_sec = remaining_steps_this_session / steps_per_sec if steps_per_sec > 0 else 0
            eta_min = eta_sec / 60.0

            print(
                f"\n[🚀 TIẾN ĐỘ - RESUME] {self.num_timesteps:,}/{self.total_timesteps_target:,} steps "
                f"({percent:.1f}%) | tốc độ: {steps_per_sec:.2f} step/s | "
                f"reward TB: {reward_str} | "
                f"ETA: ~{eta_min:.1f} phút\n", flush=True
            )
        return True


def main():
    args = parse_args()
    print("--- TIẾP TỤC HUẤN LUYỆN SAC TỪ CHECKPOINT ---")

    run_dir = args.run_dir.resolve()
    
    # train_sac.py luôn tự động lưu vào 2 file này khi bị ngắt hoặc kết thúc
    model_path = run_dir / "latest_model.zip"
    buffer_path = run_dir / "latest_replay_buffer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại '{model_path}'. "
            f"Hãy chắc chắn bạn đã truyền đúng thư mục --run-dir."
        )
    if not buffer_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy replay buffer tại '{buffer_path}'. "
            f"KHÔNG nên resume thiếu buffer này vì SAC sẽ mất toàn bộ kinh nghiệm cũ."
        )

    # Khởi tạo môi trường AgvRlEnv (Khớp với train_sac.py)
    env = AgvRlEnv(
        goals=AgvRlEnv.COMBINED_GOALS,
        goal_sampling="cycle",
        max_episode_steps=300,
    )

    # Load Model
    model = SAC.load(str(model_path), env=env)
    print(f"[resume] Đã load model — num_timesteps hiện tại: {model.num_timesteps:,}")

    # Load Buffer
    model.load_replay_buffer(str(buffer_path))
    print(f"[resume] Đã load replay buffer — số mẫu hiện có: {model.replay_buffer.size():,}")

    steps_remaining = args.total_steps - model.num_timesteps
    if steps_remaining <= 0:
        print(
            f"[resume] num_timesteps hiện tại ({model.num_timesteps:,}) đã "
            f">= TOTAL_STEPS ({args.total_steps:,}). "
            f"Không còn gì để chạy thêm."
        )
        env.close()
        return

    print(
        f"[resume] Sẽ chạy thêm {steps_remaining:,} step "
        f"(để đạt mốc tổng {args.total_steps:,})..."
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=str(run_dir),
        name_prefix='sac_vmax_checkpoint',
        save_replay_buffer=True,
    )
    progress_callback = ProgressTrackerCallback(
        total_timesteps_target=args.total_steps,
        print_freq=50
    )
    all_callbacks = CallbackList([checkpoint_callback, progress_callback])

    print("[*] (Nhấn Ctrl+C để dừng và lưu tự động)...\n")

    try:
        model.learn(
            total_timesteps=steps_remaining,
            callback=all_callbacks,
            progress_bar=False,
            log_interval=1,
            reset_num_timesteps=False, # Bắt buộc để biến đếm tiếp nối
        )

        # Lưu khi hoàn thành trọn vẹn
        model.save(str(run_dir / "final_model"))
        model.save(str(run_dir / "latest_model"))
        model.save_replay_buffer(str(run_dir / "latest_replay_buffer.pkl"))
        print(
            f"\n[*] Đã đạt mốc {args.total_steps:,} step. "
            f"Lưu thành công: final_model.zip"
        )

    except KeyboardInterrupt:
        print("\n\n[!] Phát hiện lệnh ngắt Ctrl+C từ người dùng!")
        print("[*] Đang đóng gói và lưu lại trí nhớ mạng Nơ-ron tại thời điểm hiện tại...")

        model.save(str(run_dir / "interrupted_model"))
        model.save(str(run_dir / "latest_model"))
        model.save_replay_buffer(str(run_dir / "latest_replay_buffer.pkl"))
        
        print(
            f"[*] Đã lưu thành công: interrupted_model.zip & latest_replay_buffer.pkl "
            f"(num_timesteps={model.num_timesteps:,})"
        )
        print(f"[*] Chạy lại lệnh cũ để tiếp tục từ đây.")

    finally:
        env.close()
        print("--- ĐÃ TẮT MÔI TRƯỜNG AN TOÀN ---")


if __name__ == '__main__':
    main()