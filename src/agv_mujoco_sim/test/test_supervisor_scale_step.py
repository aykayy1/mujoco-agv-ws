import pytest

from rl_velocity_supervisor import RlVelocitySupervisor


def test_fast_fall_preserves_baseline_rise_response():
    step = RlVelocitySupervisor.smoothed_scale_step
    baseline_fall = step(0.7, 0.15, 0.2, 0.25, 0.35, 0.25, 0.35)
    fast_fall = step(0.7, 0.15, 0.2, 0.25, 0.35, 1.0, 1.0)
    assert baseline_fall == pytest.approx(0.65)
    assert fast_fall == pytest.approx(0.5)
    assert step(0.3, 0.8, 0.2, 0.25, 0.35, 0.25, 0.35) == pytest.approx(
        step(0.3, 0.8, 0.2, 0.25, 0.35, 1.0, 1.0)
    )
