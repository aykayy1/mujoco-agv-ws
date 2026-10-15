"""Nav2 goal rejections should not kill a healthy training run immediately."""

import pytest

from agv_rl_env import AgvRlEnv, NavigationGoalRejected


class FakeGoalRos:
    def __init__(self, failures, error=NavigationGoalRejected):
        self.failures = failures
        self.error = error
        self.sent = []

    def send_navigation_goal(self, *goal):
        self.sent.append(goal)
        if len(self.sent) <= self.failures:
            raise self.error("rejected")


def make_env(ros, sampling="fixed"):
    env = AgvRlEnv.__new__(AgvRlEnv)
    env.ros = ros
    env.goal = (1.0, 2.0, 0.0)
    env.goal_sampling = sampling
    return env


def test_transient_rejection_retries_same_fixed_goal(monkeypatch):
    monkeypatch.setattr("agv_rl_env.time.sleep", lambda _: None)
    ros = FakeGoalRos(failures=2)
    env = make_env(ros)

    env._send_selected_goal()

    assert ros.sent == [(1.0, 2.0, 0.0)] * 3


def test_random_goal_resampled_after_rejection(monkeypatch):
    monkeypatch.setattr("agv_rl_env.time.sleep", lambda _: None)
    ros = FakeGoalRos(failures=1)
    env = make_env(ros, sampling="random_costmap")
    env._select_next_goal = lambda: setattr(env, "goal", (3.0, 4.0, 0.0))

    env._send_selected_goal()

    assert ros.sent == [(1.0, 2.0, 0.0), (3.0, 4.0, 0.0)]


def test_persistent_rejection_fails_after_five_attempts(monkeypatch):
    monkeypatch.setattr("agv_rl_env.time.sleep", lambda _: None)
    ros = FakeGoalRos(failures=9)
    env = make_env(ros)

    with pytest.raises(NavigationGoalRejected, match="5 consecutive goals"):
        env._send_selected_goal()

    assert len(ros.sent) == 5


def test_timeout_is_not_hidden_by_goal_retry():
    ros = FakeGoalRos(failures=1, error=TimeoutError)
    env = make_env(ros)

    with pytest.raises(TimeoutError):
        env._send_selected_goal()

    assert len(ros.sent) == 1
