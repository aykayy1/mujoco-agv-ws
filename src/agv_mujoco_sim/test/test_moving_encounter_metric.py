import pytest

from benchmark_dynamic_obstacle import moving_encounter_metrics


def row(time, center_distance, clearance, moving):
    return {
        "elapsed_sim_s": time,
        "obstacle_active": True,
        "obstacle_moving": moving,
        "obstacle_center_distance_m": center_distance,
        "estimated_body_clearance_m": clearance,
        "obstacle_y": 0.0,
    }


def test_crossing_counts_when_later_parked_obstacle_is_closer():
    metrics = moving_encounter_metrics([
        row(5.0, 1.4, 0.3, True),
        row(20.0, 1.0, 0.1, False),
    ])
    assert metrics["moving_at_closest_approach"] is False
    assert metrics["moving_encounter"] is True
    assert metrics["moving_closest_approach_time_s"] == 5.0
    assert metrics["minimum_moving_estimated_body_clearance_m"] == pytest.approx(0.3)


def test_no_moving_encounter_when_crossing_remains_far():
    metrics = moving_encounter_metrics([
        row(5.0, 2.2, 1.0, True),
        row(20.0, 1.0, 0.1, False),
    ])
    assert metrics["moving_encounter"] is False


def test_no_obstacle_returns_null_distances():
    metrics = moving_encounter_metrics([])
    assert metrics["moving_encounter"] is False
    assert metrics["minimum_moving_obstacle_center_distance_m"] is None
