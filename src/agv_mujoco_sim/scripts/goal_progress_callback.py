"""Custom SB3 callback logging continuing-task navigation stats to TensorBoard.

Vì AgvRlEnv là "continuing task" (robot có thể tới nhiều goal trong 1
episode), success_rate nhị phân (0/1) mặc định của SB3 không đủ để đánh giá
chất lượng chính sách. Callback này bổ sung các chỉ số liên tục hơn:

    custom/goals_reached_mean    - số goal trung bình đạt được mỗi episode
                                    kết thúc (trong cửa sổ window_size gần
                                    nhất)
    custom/goals_reached_max     - số goal nhiều nhất đạt được trong 1
                                    episode (trong cùng cửa sổ)
    custom/distance_to_goal_mean - khoảng cách trung bình tới goal tại thời
                                    điểm episode kết thúc
    custom/collision_rate        - tỉ lệ episode kết thúc do va chạm

Các chỉ số built-in của SB3 (rollout/success_rate, rollout/ep_rew_mean,
rollout/ep_len_mean) vẫn tiếp tục hoạt động bình thường song song, không bị
ảnh hưởng bởi callback này.
"""

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GoalProgressLoggingCallback(BaseCallback):
    """Đọc info dict mỗi khi 1 episode kết thúc và ghi thống kê lên TensorBoard."""

    def __init__(self, window_size: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window_size = window_size
        self._goals_reached: deque = deque(maxlen=window_size)
        self._distance_to_goal: deque = deque(maxlen=window_size)
        self._collision_flags: deque = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        # self.locals được SB3 điền tự động sau mỗi lần gọi vec_env.step()
        # bất kể n_envs = 1 hay nhiều, luôn là list/array theo từng env con.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if dones is None:
            # Một số phiên bản SB3/thuật toán dùng key khác, phòng hờ.
            dones = self.locals.get("terminated", [False] * len(infos))

        for env_idx, info in enumerate(infos):
            is_done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if not is_done:
                continue

            # Khi VecEnv tự auto-reset lúc done=True, "info" ở bước đó vẫn
            # là info của BƯỚC KẾT THÚC (trước khi reset), nên các key custom
            # ta set trong step() (goals_reached_this_episode, v.v.) vẫn còn
            # nguyên - không bị ghi đè bởi info của lần reset kế tiếp.
            if "goals_reached_this_episode" in info:
                self._goals_reached.append(
                    float(info["goals_reached_this_episode"])
                )
            if "distance_to_goal" in info:
                self._distance_to_goal.append(float(info["distance_to_goal"]))
            if "termination_reason" in info:
                self._collision_flags.append(
                    1.0 if info["termination_reason"] == "collision" else 0.0
                )

        if len(self._goals_reached) > 0:
            self.logger.record(
                "custom/goals_reached_mean", float(np.mean(self._goals_reached))
            )
            self.logger.record(
                "custom/goals_reached_max", float(np.max(self._goals_reached))
            )
        if len(self._distance_to_goal) > 0:
            self.logger.record(
                "custom/distance_to_goal_mean",
                float(np.mean(self._distance_to_goal)),
            )
        if len(self._collision_flags) > 0:
            self.logger.record(
                "custom/collision_rate", float(np.mean(self._collision_flags))
            )

        return True