"""Phase 0 configuration tests."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from uav_sim.config import AircraftConfig, config_hash, load_aircraft

CONFIG_PATH = Path("config/aircraft_v1.yaml")


def _load_raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_load_aircraft_returns_frozen_config():
    cfg = load_aircraft(CONFIG_PATH)

    assert isinstance(cfg, AircraftConfig)
    assert cfg.name == "aerosonde_uav_v1"
    with pytest.raises(ValidationError):
        cfg.name = "changed"


def test_unstable_pitch_sign_is_rejected():
    raw = _load_raw_config()
    raw["aero"]["C_m_alpha"] = 0.38

    with pytest.raises(ValidationError, match="C_m_alpha.*sign convention"):
        AircraftConfig.model_validate(raw)


def test_missing_required_key_is_reported():
    raw = _load_raw_config()
    del raw["mass"]["m"]

    with pytest.raises(ValidationError, match="mass.*m"):
        AircraftConfig.model_validate(raw)


def test_loaded_angle_fields_are_radians_only():
    cfg = load_aircraft(CONFIG_PATH)
    dumped = cfg.model_dump(mode="json")

    assert cfg.actuators.aileron.max_rad == pytest.approx(math.radians(25.0))
    assert cfg.envelope.alpha_max_rad == pytest.approx(math.radians(12.0))
    assert all(not key.endswith("_deg") for key in _all_keys(dumped))
    assert all(not key.endswith("_dps") for key in _all_keys(dumped))


def test_inertia_tensor_is_symmetric_positive_definite():
    cfg = load_aircraft(CONFIG_PATH)
    inertia = cfg.mass.inertia_tensor

    assert inertia.shape == (3, 3)
    assert np.allclose(inertia, inertia.T)
    assert np.all(np.linalg.eigvalsh(inertia) > 0.0)
    assert np.allclose(cfg.mass.inertia_inverse @ inertia, np.eye(3))


def test_config_hash_is_stable_and_sensitive_to_changes():
    cfg = load_aircraft(CONFIG_PATH)
    copied = AircraftConfig.model_validate(copy.deepcopy(cfg.model_dump(mode="json")))
    changed = cfg.model_copy(update={"name": "aerosonde_uav_v1_changed"})

    assert config_hash(cfg) == config_hash(copied)
    assert config_hash(cfg) != config_hash(changed)


def test_benchmark_control_signs_match_conventions():
    cfg = load_aircraft(CONFIG_PATH)

    assert cfg.aero.C_m_alpha < 0.0
    assert cfg.aero.C_m_q < 0.0
    assert cfg.aero.C_l_p < 0.0
    assert cfg.aero.C_n_r < 0.0
    assert cfg.aero.C_L_alpha > 0.0
    assert cfg.aero.C_l_delta_a > 0.0
    assert cfg.aero.C_n_delta_r > 0.0
