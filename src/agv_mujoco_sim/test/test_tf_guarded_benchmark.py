"""The benchmark must not count a trial after controller TF freezes."""

import json

import pytest

from run_mppi_three_maps import TfStaleMonitor, load_replay_scenarios


def tf_line(wall_time, data_time, transform_time):
    return (
        f"[controller_server-7] [ERROR] [{wall_time:.3f}] [tf_help]: "
        f"Data time: {data_time}s 0ns, "
        f"Transform time: {transform_time}s 0ns\n"
    )


def test_single_or_small_tf_lag_does_not_invalidate_trial(tmp_path):
    log = tmp_path / "launch.log"
    log.write_text(tf_line(100.0, 102, 100))
    monitor = TfStaleMonitor(log)
    assert monitor.poll() is None
    with log.open("a") as stream:
        stream.write("".join(tf_line(100.1 + i * 0.1, 101, 100) for i in range(8)))
    assert monitor.poll() is None


def test_persistent_stale_controller_tf_invalidates_trial(tmp_path):
    log = tmp_path / "launch.log"
    log.write_text("".join(
        tf_line(100.0 + i * 0.1, 103 + i, 100) for i in range(8)
    ))
    fault = TfStaleMonitor(log).poll()
    assert fault["reason"] == "controller_tf_stale"
    assert fault["lag_s"] >= 2.0


def test_monitor_ignores_previous_trial_log(tmp_path):
    log = tmp_path / "launch.log"
    log.write_text("".join(tf_line(100.0 + i * 0.1, 103 + i, 100) for i in range(8)))
    monitor = TfStaleMonitor(log, log.stat().st_size)
    assert monitor.poll() is None
    with log.open("a") as stream:
        stream.write(tf_line(101.0, 102, 101))
    assert monitor.poll() is None


def test_replay_uses_exact_original_scenarios_and_order(tmp_path):
    map_dir = tmp_path / "mppi_warehouse_aisles"
    map_dir.mkdir()
    scenarios = [
        {"trigger_x": 2.2, "obstacle_x": 5.1, "obstacle_start_y": -1.2,
         "obstacle_end_y": 1.3, "obstacle_speed_m_s": 0.3},
        {"trigger_x": 2.4, "obstacle_x": 4.9, "obstacle_start_y": 1.1,
         "obstacle_end_y": -1.4, "obstacle_speed_m_s": 0.4},
    ]
    report = map_dir / "benchmark.json"
    report.write_text(json.dumps({"trials": [
        {"trial": 2, "scenario": scenarios[0]},
        {"trial": 94, "scenario": scenarios[1]},
    ]}))
    assert load_replay_scenarios(report, [94, 2], map_dir.name) == scenarios[::-1]
    with pytest.raises(ValueError, match="absent"):
        load_replay_scenarios(report, [100], map_dir.name)
    with pytest.raises(ValueError, match="must be inside"):
        load_replay_scenarios(report, [2], "mppi_warehouse_mixed")
