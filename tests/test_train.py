"""Unit tests for train.py argument parsing, callback, and components."""

import argparse
from collections import deque
import numpy as np
import pytest

from train import FlagOrValueAction, LandingStatsCallback, parse_args


def test_parse_args_defaults():
    """Verify default arguments match user specifications."""
    args = parse_args([])
    assert args.seed == 42
    assert args.total_steps == 1_000_000
    assert args.n_envs == 8
    assert args.vertical_only is False
    assert args.wind is False
    assert args.noise is False
    assert args.randomize is False
    assert args.run_name is None


def test_parse_args_custom_values():
    """Verify custom flags and values are parsed accurately."""
    args = parse_args([
        "--seed", "123",
        "--total-steps", "50000",
        "--n-envs", "4",
        "--vertical-only",
        "--wind", "250.0",
        "--noise", "0.05",
        "--randomize",
        "--run-name", "test_smoke_run",
    ])
    assert args.seed == 123
    assert args.total_steps == 50000
    assert args.n_envs == 4
    assert args.vertical_only is True
    assert args.wind == 250.0
    assert args.noise == 0.05
    assert args.randomize is True
    assert args.run_name == "test_smoke_run"


def test_flag_or_value_action_flags():
    """Verify FlagOrValueAction behaves as boolean flags when no value is provided."""
    args = parse_args(["--wind", "--noise", "--randomize"])
    assert args.wind is True
    assert args.noise is True
    assert args.randomize is True


def test_landing_stats_callback():
    """Verify LandingStatsCallback accurately computes rolling success rate and speeds."""
    cb = LandingStatsCallback(window_size=100)

    class DummyLogger:
        def __init__(self):
            self.logged = {}

        def record(self, key, val):
            self.logged[key] = val

    class DummyModel:
        def __init__(self, logger):
            self.logger = logger

    dummy_logger = DummyLogger()
    cb.model = DummyModel(dummy_logger)

    # Simulate 5 steps with dones
    # Episode 1: landed, vy = -1.2
    # Episode 2: crashed, vy = -15.0
    # Episode 3: landed, vy = -0.8
    cb.locals = {
        "dones": [True, True, True],
        "infos": [
            {"landed": True, "touchdown_vy": -1.2},
            {"landed": False, "touchdown_vy": -15.0},
            {"landed": True, "touchdown_vy": -0.8},
        ],
    }
    cb._on_step()

    assert cb.total_episodes == 3
    assert cb.total_landings == 2
    assert len(cb.landed_history) == 3
    assert np.mean(cb.landed_history) == pytest.approx(2.0 / 3.0)
    assert len(cb.touchdown_speed_history) == 3
    assert np.mean(cb.touchdown_speed_history) == pytest.approx((1.2 + 15.0 + 0.8) / 3.0)
    assert dummy_logger.logged["landing/success_rate"] == pytest.approx(2.0 / 3.0)
    assert dummy_logger.logged["landing/mean_touchdown_speed"] == pytest.approx((1.2 + 15.0 + 0.8) / 3.0)
