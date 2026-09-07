"""Contract + regression tests for the multi-seed analysis (analysis/compare_seeds.py)
and the shared GPU measurement helpers (utils/efficiency.py).

Same conventions as test_pipeline.py: dependency-light, plain assertions, runs
either under pytest or directly with `python tests/test_analysis_regressions.py`.

    python -m pytest tests/test_analysis_regressions.py -q
    python tests/test_analysis_regressions.py
"""

import csv
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import analysis.compare_seeds as compare_seeds
from utils import efficiency
from utils.io_helpers import stage2_dir


# --------------------------------------------------------------------------- #
# analysis/compare_seeds.py: corpus_wer()
# --------------------------------------------------------------------------- #
def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "reference", "hypothesis", "wer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})


def test_corpus_wer_basic():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.csv")
        # 10-word reference, wer=0.20 -> 2.0 errors -> corpus WER 20.0%
        _write_csv(p, [_fixture_row(10, 0.20)])
        assert abs(compare_seeds.corpus_wer(p) - 20.0) < 1e-9


def test_corpus_wer_missing_file_returns_none():
    assert compare_seeds.corpus_wer("/no/such/path/wer_x.csv") is None


def test_corpus_wer_empty_file_returns_none():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "empty.csv")
        _write_csv(p, [])
        assert compare_seeds.corpus_wer(p) is None


# --------------------------------------------------------------------------- #
# analysis/compare_seeds.py: build_rows() aggregation math, via a synthetic
# stage2_processed/ tree (monkeypatches the bare `stage2_dir`/`analysis_dir`
# names compare_seeds imported, same trick needed because the production code
# is filesystem-coupled by design).
# --------------------------------------------------------------------------- #
def _fixture_row(ref_words: int, wer: float) -> dict:
    words = [f"w{i}" for i in range(ref_words)]
    errors = int(round(ref_words * wer))
    return {
        "ID": "clip-1",
        "reference": " ".join(words),
        "hypothesis": " ".join(words[:ref_words - errors]),
        "wer": str(wer),
    }


def test_build_rows_seed_aggregation_and_guards():
    tmp = tempfile.mkdtemp()
    orig_stage2, orig_analysis = compare_seeds.stage2_dir, compare_seeds.analysis_dir
    try:
        s2 = os.path.join(tmp, "stage2_processed")
        an = os.path.join(tmp, "analysis")
        os.makedirs(os.path.join(s2, "transcript_clean"), exist_ok=True)
        os.makedirs(an, exist_ok=True)
        compare_seeds.stage2_dir = lambda dataset="tie", _s2=s2: _s2
        compare_seeds.analysis_dir = lambda dataset="tie", _an=an: _an

        ds, mode = "fixtureds", "transcript_clean"

        # --- tiny: baseline present, 3 clean seeds + 1 unreadable seed ------
        _write_csv(os.path.join(s2, mode, "wer_tiny_hf_transcript_clean.csv"),
                   [_fixture_row(100, 0.30)])                      # baseline 30.0%
        _write_csv(os.path.join(s2, mode, f"wer_tiny_{ds}_ft_seed42_transcript_clean.csv"),
                   [_fixture_row(100, 0.20)])                      # 20.0% -> delta -10.0
        _write_csv(os.path.join(s2, mode, f"wer_tiny_{ds}_ft_seed43_transcript_clean.csv"),
                   [_fixture_row(100, 0.22)])                      # 22.0% -> delta -8.0
        _write_csv(os.path.join(s2, mode, f"wer_tiny_{ds}_ft_seed44_transcript_clean.csv"),
                   [_fixture_row(100, 0.24)])                      # 24.0% -> delta -6.0
        _write_csv(os.path.join(s2, mode, f"wer_tiny_{ds}_ft_seed45_transcript_clean.csv"), [])  # unreadable

        # --- small: seeds present, NO baseline file at all ------------------
        _write_csv(os.path.join(s2, mode, f"wer_small_{ds}_ft_seed42_transcript_clean.csv"),
                   [_fixture_row(100, 0.10)])
        _write_csv(os.path.join(s2, mode, f"wer_small_{ds}_ft_seed43_transcript_clean.csv"),
                   [_fixture_row(100, 0.14)])

        rows, per_seed = compare_seeds.build_rows(ds, mode)
        by_size = {r["size"]: r for r in rows}

        tiny = by_size["tiny"]
        # The 4th seed file exists on disk but is unreadable: must be EXCLUDED from
        # n_seeds/seeds (the bug this session fixed), not silently included as a zero.
        assert tiny["n_seeds"] == 3, tiny
        assert tiny["seeds"] == "42,43,44", tiny
        assert abs(tiny["hf_baseline_wer"] - 30.0) < 1e-6
        assert abs(tiny["delta_pp_mean"] - (-8.0)) < 1e-6
        assert abs(tiny["delta_pp_sd"] - 2.0) < 1e-6
        assert abs(tiny["delta_pp_min"] - (-10.0)) < 1e-6
        assert abs(tiny["delta_pp_max"] - (-6.0)) < 1e-6

        small = by_size["small"]
        # No baseline -> `deltas` is empty and `arr` falls back to absolute WERs.
        # delta_pp_sd must be None here (the exact regression this session fixed):
        # publishing the spread of absolute WERs under a column documented as a
        # delta spread would be wrong even though arr.std() is well-defined.
        assert small["hf_baseline_wer"] is None
        assert small["delta_pp_mean"] is None
        assert small["delta_pp_sd"] is None
        assert small["delta_pp_min"] is None
        assert small["delta_pp_max"] is None
        assert small["n_seeds"] == 2

        assert "medium" not in by_size  # no files at all -> skipped entirely

        # per-seed rows: one per run that actually contributed, so the aggregate above
        # is auditable. The unreadable 4th tiny seed must be absent here too, and the
        # baseline-less small rows must carry delta_pp=None rather than a bare WER.
        assert [(r["size"], r["seed"]) for r in per_seed] == [
            ("tiny", 42), ("tiny", 43), ("tiny", 44), ("small", 42), ("small", 43)]
        assert sorted(r["delta_pp"] for r in per_seed if r["size"] == "tiny") == [-10.0, -8.0, -6.0]
        assert all(r["delta_pp"] is None and r["hf_baseline_wer"] is None
                   for r in per_seed if r["size"] == "small")
    finally:
        compare_seeds.stage2_dir = orig_stage2
        compare_seeds.analysis_dir = orig_analysis
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Regression gate: committed 6-seed AESRC Tiny data (mirrors test_pipeline.py's
# committed-numbers pattern). Skips gracefully if the data isn't present.
# --------------------------------------------------------------------------- #
def test_committed_aesrc_tiny_seeds_regression():
    p = os.path.join(stage2_dir("aesrc"), "transcript_clean")
    if not os.path.isdir(p):
        pytest.skip("no committed AESRC stage2 output")
    rows, per_seed = compare_seeds.build_rows("aesrc", "transcript_clean")
    tiny = next((r for r in rows if r["size"] == "tiny"), None)
    if tiny is None:
        pytest.skip("no committed AESRC tiny multi-seed data")
    assert tiny["n_seeds"] == 6
    assert abs(tiny["delta_pp_mean"] - (-6.85)) < 0.01
    assert abs(tiny["delta_pp_sd"] - 1.03) < 0.01
    assert abs(tiny["delta_pp_min"] - (-7.337)) < 0.01
    assert abs(tiny["delta_pp_max"] - (-4.751)) < 0.01

    # The paper claims every one of the 18 runs improves on its own baseline. That is a
    # per-run claim, so gate it on the per-run data rather than on min/max alone.
    assert len(per_seed) == 18, len(per_seed)
    assert all(r["delta_pp"] < 0 for r in per_seed), \
        [r for r in per_seed if r["delta_pp"] >= 0]
    # Tiny's SD of 1.03 comes from a single anomalous seed, not broad spread: the other
    # five sit inside a 0.12 pp band. The paper's discussion depends on this, so pin it.
    tiny_deltas = sorted(r["delta_pp"] for r in per_seed if r["size"] == "tiny")
    assert abs(tiny_deltas[0] - (-7.337)) < 0.01 and abs(tiny_deltas[-1] - (-4.751)) < 0.01
    assert abs(tiny_deltas[-2] - tiny_deltas[0]) < 0.15, tiny_deltas


# --------------------------------------------------------------------------- #
# utils/efficiency.py: pure helpers, no GPU needed.
# --------------------------------------------------------------------------- #
def test_subset_fingerprint_deterministic_and_sensitive():
    ids = ["clip_a", "clip_b", "clip_c"]
    fp1 = efficiency.subset_fingerprint(ids)
    fp2 = efficiency.subset_fingerprint(list(ids))
    assert fp1 == fp2 and len(fp1) == 12
    assert efficiency.subset_fingerprint(ids[:-1]) != fp1


def test_subset_selection_rejects_empty_workloads():
    class Empty:
        def __len__(self):
            return 0

    for ds, n in ((Empty(), 1), (object(), 0)):
        try:
            efficiency.select_subset(ds, n, 42)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid empty benchmark workload was accepted")


def test_count_parameters_shapes():
    class Flat:
        def parameters(self):
            return [type("P", (), {"numel": lambda self: 3})()] * 4  # 12 params

    class Wrapped:
        def __init__(self):
            self.model = Flat()

    class NoParams:
        pass

    assert efficiency.count_parameters(Flat()) == 12
    assert efficiency.count_parameters(Wrapped()) == 12
    assert efficiency.count_parameters(NoParams()) is None


def test_hardware_provenance_always_has_base_keys():
    info = efficiency.hardware_provenance()
    for k in ("hostname", "platform", "python", "cpu_count", "device"):
        assert k in info


def test_cudnn_disabled_restores_previous_setting():
    torch = pytest.importorskip("torch")
    before = torch.backends.cudnn.enabled
    with efficiency.cudnn_disabled():
        assert torch.backends.cudnn.enabled is False
    assert torch.backends.cudnn.enabled == before


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
