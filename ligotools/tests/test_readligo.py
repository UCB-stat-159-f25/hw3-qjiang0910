from pathlib import Path
import json
import numpy as np
import pytest

from ligotools import readligo as rl

# repo_root/data
DATA = Path(__file__).resolve().parents[2] / "data"

def _events():
    fnjson = DATA / "BBH_events_v3.json"
    with open(fnjson, "r") as f:
        return json.load(f)

@pytest.mark.parametrize("ifo_key,ifo_label", [
    ("fn_H1", "H1"),
    ("fn_L1", "L1"),
])
def test_loaddata_time_and_strain_are_aligned_and_monotonic(ifo_key, ifo_label):
    """
    Checks strain and time have the same length and time goes up steadily.
    """
    events = _events()
    ev = events["GW150914"]  # canonical tutorial event
    fpath = DATA / ev[ifo_key]
    assert fpath.exists(), f"Missing test data file: {fpath}"

    strain, time, chan = rl.loaddata(str(fpath), ifo_label, tvec=True, readstrain=True)

    # basic alignment and sanity
    assert len(strain) == len(time) and len(time) > 0
    assert np.isfinite(strain).all()
    assert np.isfinite(time).all()
    # time strictly increasing
    dt = np.diff(time)
    assert np.all(dt > 0), "time must be strictly increasing"
    # channel dict should exist and contain something (e.g., DEFAULT if available)
    assert isinstance(chan, dict)

@pytest.mark.parametrize("ifo_key,ifo_label", [
    ("fn_H1", "H1"),
    ("fn_L1", "L1"),
])
def test_loaddata_meta_duration_matches_dq_lengths(ifo_key, ifo_label):
    """
    Checks that 1 Hz data length matches how long the recording lasts.
    """
    events = _events()
    ev = events["GW150914"]
    fpath = DATA / ev[ifo_key]
    assert fpath.exists(), f"Missing test data file: {fpath}"

    strain, meta, chan = rl.loaddata(str(fpath), ifo_label, tvec=False, readstrain=True)
    assert isinstance(meta, dict)
    assert {"start", "stop", "dt"} <= set(meta.keys())
    duration = int(meta["stop"] - meta["start"])
    assert duration > 0

    # pick a representative channel if DEFAULT exists; otherwise use any present key
    ch_key = "DEFAULT" if "DEFAULT" in chan else next(iter(chan.keys()))
    dq = chan[ch_key]
    assert dq.ndim == 1
    assert len(dq) == duration, "1 Hz DQ length must equal number of seconds in meta duration"

def test_json_filenames_exist_in_data_folder():
    """
    Checks that all files listed in the JSON actually exist in data/.
    """
    events = _events()
    ev = events["GW150914"]
    for k in ("fn_H1", "fn_L1", "fn_template"):
        p = DATA / ev[k]
        assert p.exists(), f"Expected file missing: {p}"