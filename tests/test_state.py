"""State layout and accessor tests."""

import re
from pathlib import Path

import numpy as np
import pytest

from uav_sim.rotations import euler_to_quat
from uav_sim.state import State, initial_state, make_state


def test_state_round_trip_returns_array_exactly():
    rng = np.random.default_rng(3)
    x = rng.normal(size=13)
    np.testing.assert_array_equal(State.from_array(x).to_array(), x)


def test_altitude_uses_negative_down_position():
    x = make_state(
        pos_n=[0.0, 0.0, -100.0],
        vel_b=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[0.0, 0.0, 0.0],
    )
    assert State.from_array(x).altitude == 100.0


def test_make_state_rejects_non_unit_quaternion():
    with pytest.raises(ValueError, match="unit norm"):
        make_state(
            pos_n=[0.0, 0.0, 0.0],
            vel_b=[0.0, 0.0, 0.0],
            quat=[2.0, 0.0, 0.0, 0.0],
            omega_b=[0.0, 0.0, 0.0],
        )


def test_initial_state_uses_altitude_and_alpha():
    x = initial_state(V=25.0, alpha=0.1, h=123.0)
    state = State.from_array(x)

    assert state.altitude == 123.0
    assert state.vel_b[0] == pytest.approx(25.0 * np.cos(0.1))
    assert state.vel_b[2] == pytest.approx(25.0 * np.sin(0.1))
    np.testing.assert_allclose(state.quat, euler_to_quat(0.0, 0.1, 0.0))


def test_no_raw_state_slice_indexing_outside_state_module():
    pattern = re.compile(r"x\[\s*\d+\s*:\s*\d+\s*\]")
    for path in Path("src/uav_sim").rglob("*.py"):
        if path.name == "state.py":
            continue
        assert not pattern.search(path.read_text(encoding="utf-8")), path
