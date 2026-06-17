"""Validate the TLE cross-match logic in space_oracle (offline, no network).

We stub the Celestrak download, inject a single real ISS TLE, freeze time, and put
the observer at the satellite's sub-point so it is overhead — a deterministic
geometry to exercise find_match() without depending on the wall clock or the net.
"""

from __future__ import annotations

import os
import sys

import pytest

skyfield = pytest.importorskip("skyfield")
from skyfield.api import EarthSatellite, wgs84  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from space_oracle import SpaceOracle  # noqa: E402

# Canonical ISS TLE from the Skyfield documentation (valid format + checksums).
ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


@pytest.fixture
def oracle(monkeypatch, tmp_path):
    # never hit the network / cache
    monkeypatch.setattr(SpaceOracle, "_load_or_download_tles", lambda self: None)
    o = SpaceOracle(cache_file=str(tmp_path / "none.txt"))
    o.satellites = [EarthSatellite(ISS_L1, ISS_L2, "ISS (ZARYA)", o.ts)]
    # freeze time so find_match() (which calls ts.now()) is deterministic
    fixed_t = o.ts.utc(2024, 1, 1, 12, 0, 0)
    monkeypatch.setattr(o.ts, "now", lambda: fixed_t)
    o._fixed_t = fixed_t
    return o


def _overhead_azalt(o):
    """Place the observer at the ISS sub-point so the sat is ~overhead; return az/alt."""
    sat = o.satellites[0]
    sub = wgs84.subpoint(sat.at(o._fixed_t))
    o.update_observer(sub.latitude.degrees, sub.longitude.degrees)
    alt, az, _ = (sat - o.observer).at(o._fixed_t).altaz()
    return az.degrees, alt.degrees


def test_catalogue_loaded(oracle):
    assert len(oracle.satellites) == 1


def test_match_at_satellite_position(oracle):
    az, alt = _overhead_azalt(oracle)
    assert alt > 80.0                       # forced overhead by sub-point geometry
    name, dist = oracle.find_match(az, alt, tolerance_deg=2.0)
    assert name == "ISS (ZARYA)"
    assert dist < 0.5


def test_no_match_far_from_any_satellite(oracle):
    az, _alt = _overhead_azalt(oracle)
    # query a wildly different elevation: the only sat is overhead, nothing near alt=20
    name, dist = oracle.find_match((az + 90) % 360, 20.0, tolerance_deg=2.0)
    assert name is None and dist is None


def test_below_horizon_satellites_are_ignored(oracle):
    # observer on the opposite side of the planet -> ISS below horizon -> no match
    sat = oracle.satellites[0]
    sub = wgs84.subpoint(sat.at(oracle._fixed_t))
    oracle.update_observer(-sub.latitude.degrees, (sub.longitude.degrees + 180) % 360 - 180)
    name, _ = oracle.find_match(0.0, 45.0, tolerance_deg=5.0)
    assert name is None


def test_empty_catalogue_returns_none(oracle):
    oracle.satellites = []
    assert oracle.find_match(180.0, 45.0) == (None, None)


def test_update_observer(oracle):
    oracle.update_observer(0.0, 0.0)
    assert abs(oracle.observer.latitude.degrees) < 1e-6
