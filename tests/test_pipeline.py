"""Core contract + regression tests for the benchmark pipeline.

Deliberately dependency-light: needs only the analysis stack already in
requirements.txt (jiwer, num2words, whisper_normalizer, pandas). Run either way:

    python -m pytest tests/ -q
    python tests/test_pipeline.py          # plain-python fallback, no pytest needed

The point is to pin the invariants that, if they silently drifted, would corrupt
every reported number: the normalization contract, WER/CER edge-case accounting,
registry integrity, and the committed headline corpus-WER values (regression gate).
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import registry
from utils.normalize import (
    normalize_text, minimal_clean_text, whisper_normalize_text, normalize_for_mode,
)
from utils.wer_compute import (
    compute_sample_wer, compute_corpus_wer, compute_corpus_cer,
    reference_word_recall, length_ratio,
)
from utils.io_helpers import positive_float, stage2_dir, text_value
from normalize_and_score import process

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- #
# Normalization contract (applied symmetrically to ref + hyp, must not drift).
# --------------------------------------------------------------------------- #
def test_custom_normalizer_contract():
    assert normalize_text("1st") == "first"                # ordinal -> word
    assert normalize_text("100") == "one hundred"          # cardinal -> words
    assert normalize_text("3.5") == "three point five"     # decimal
    assert normalize_text("don't") == "dont"               # contraction: apostrophe stripped, not expanded
    assert normalize_text("Bernoulli's") == "bernoulli s"  # possessive split
    assert normalize_text("") == "" and normalize_text(None or "") == ""
    assert normalize_text(float("nan")) == ""


def test_minimal_normalizer_contract():
    assert minimal_clean_text('"Hello, World!"') == "hello world"   # strip quotes/punct/case
    assert minimal_clean_text("1st") == "1st"                        # NO number expansion
    assert minimal_clean_text("") == ""
    assert minimal_clean_text(float("nan")) == ""


def test_missing_text_values_never_become_literal_nan():
    assert text_value(None) == ""
    assert text_value(float("nan")) == ""
    assert text_value("  hello  ") == "hello"
    assert positive_float(1.5) == 1.5
    assert positive_float(float("nan")) is None
    assert positive_float(0) is None


def test_scoring_preserves_an_empty_hypothesis_as_deletions():
    df = pd.DataFrame({
        "ID": ["clip-1"],
        "transcript_raw": ["hello world"],
        "normalised_transcript_raw": [""],
        "hypothesis_raw": [float("nan")],
    })
    rows, stats = process(df, "tiny", "transcript_clean")
    assert rows[0]["hypothesis_raw"] == ""
    assert rows[0]["hypothesis"] == ""
    assert stats["num_empty_hyps"] == 1
    assert stats["deletions"] == 2


def test_whisper_normalizer_contract():
    # Community-standard Whisper normalizer DOES expand contractions.
    assert whisper_normalize_text("I don't") == "i do not"
    assert whisper_normalize_text("") == ""


def test_mode_dispatch_matches_registry():
    # normalize_for_mode must route each mode to the normalizer the registry declares.
    assert normalize_for_mode("transcript_clean", "1st") == "first"      # custom
    assert normalize_for_mode("transcript_raw", "1st") == "1st"          # minimal
    assert normalize_for_mode("whisper_norm", "don't") == "do not"       # whisper


# --------------------------------------------------------------------------- #
# WER / CER accounting edge cases.
# --------------------------------------------------------------------------- #
def test_wer_edge_cases():
    assert compute_sample_wer("a b c", "") == 1.0     # empty hyp = all deletions
    assert compute_sample_wer("", "x") == 0.0         # empty ref = 0
    assert compute_sample_wer("a b", "a b") == 0.0    # exact match

def test_corpus_wer_insertions_can_exceed_100pct():
    r = compute_corpus_wer(["a"], ["a b c"])          # 2 insertions on a 1-word ref
    assert r["corpus_wer"] == 2.0
    assert r["insertions"] == 2 and r["insertion_rate"] == 2.0

def test_corpus_wer_empty_hyp_counts_as_deletions():
    r = compute_corpus_wer(["a b c", "x y"], ["", "x y"])
    assert r["num_empty_hyps"] == 1
    assert r["deletions"] >= 3                        # the 3 words of the empty-hyp ref

def test_corpus_metrics_reject_misaligned_inputs():
    for scorer in (compute_corpus_wer, compute_corpus_cer):
        try:
            scorer(["one", "two"], ["one"])
        except ValueError as exc:
            assert "count mismatch" in str(exc)
        else:
            raise AssertionError("misaligned corpus inputs were silently truncated")

def test_corpus_cer_and_diagnostics():
    assert compute_corpus_cer(["abc"], [""]) == 1.0   # empty hyp = all chars deleted
    assert reference_word_recall("a b c", "a b x") == 2 / 3
    assert length_ratio("a b", "a b c d") == 2.0
    assert length_ratio("", "x") == 0.0               # guarded, no div-by-zero


# --------------------------------------------------------------------------- #
# Registry integrity, the single source of truth must stay self-consistent.
# --------------------------------------------------------------------------- #
def test_registry_integrity():
    assert registry.PRIMARY_MODE in registry.ALL_MODES
    keys = [m.key for m in registry.MODEL_SPECS]
    assert len(keys) == len(set(keys)), "duplicate model keys"
    orders = [m.order for m in registry.MODEL_SPECS]
    assert len(orders) == len(set(orders)), "duplicate sort orders"
    # Chart models share figures, so their colours must be distinct. The FT-study
    # variants (chart=False) deliberately reuse Medium's hue family and are excluded.
    chart_colors = [registry.MODEL_BY_KEY[k].color for k in registry.CHART_MODELS]
    assert len(chart_colors) == len(set(chart_colors)), "duplicate colours among chart models"
    for m in registry.MODEL_SPECS:
        assert m.engine in registry.ENGINES
        assert m.arch_class in registry.ARCH_CLASSES
        if m.only_datasets is not None:
            assert set(m.only_datasets) <= set(registry.DATASET_BY_KEY), \
                f"{m.key}.only_datasets references an unknown dataset"
    # Svarah is eval-only: no FT variants or HF baselines apply there.
    assert set(registry.models_for_dataset("svarah")).isdisjoint(
        {m.key for m in registry.MODEL_SPECS if m.only_datasets is not None})
    assert "hf_raw" not in registry.modes_for_dataset("svarah")   # Svarah has no alt reference


def test_dataset_model_gating():
    tie_models = set(registry.models_for_dataset("tie"))
    aesrc_models = set(registry.models_for_dataset("aesrc"))
    # TIE keeps its FT variants; AESRC gets its own plus the shared HF baselines.
    assert {"tiny_ft", "small_ft", "medium_ft"} <= tie_models
    assert {"tiny_ft", "small_ft", "medium_ft"}.isdisjoint(aesrc_models)
    assert {"tiny_aesrc_ft", "small_aesrc_ft", "medium_aesrc_ft"} <= aesrc_models
    assert {"tiny_aesrc_ft", "small_aesrc_ft", "medium_aesrc_ft"}.isdisjoint(tie_models)
    assert {"tiny_hf", "small_hf", "medium_hf"} <= tie_models & aesrc_models
    # All 9 pretrained chart models apply everywhere.
    for d in registry.DATASET_BY_KEY:
        assert set(registry.CHART_MODELS) <= set(registry.models_for_dataset(d))


def test_aesrc_spec():
    spec = registry.get_dataset("aesrc")
    assert spec.filter_col == "accent" and spec.filter_value == "INDIAN"
    assert spec.audio_undecoded is True          # bytes-stored audio needs the decode bypass
    assert spec.duration_col is None             # duration derived from audio bytes
    assert spec.speaker_col == "speaker"
    assert "hf_raw" not in spec.applicable_modes  # no alt reference field
    assert set(spec.splits) == {"train", "validation", "eval"}


def test_dataset_modes_reference_valid_roles():
    for d in registry.DATASET_SPECS:
        for mode in d.applicable_modes:
            role = registry.get_reference_role(mode)
            assert role in ("gold", "alt")
            if role == "alt":
                assert d.alt_ref_col is not None, f"{d.key} uses alt-ref mode {mode} but has no alt_ref_col"


# --------------------------------------------------------------------------- #
# Regression gate, committed headline corpus-WER values (transcript_clean).
# Skips gracefully if Stage 2 outputs aren't present.
# --------------------------------------------------------------------------- #
EXPECTED_TIE_WER = {
    "base": 17.53, "medium": 14.76, "large": 15.93, "parakeet": 15.60, "qwen3": 16.66,
    "tiny": 19.43, "tiny_hf": 22.10, "tiny_ft": 19.14,
    "small": 16.05, "small_hf": 17.38, "small_ft": 16.21,
    "large_v3_turbo": 17.98, "parakeet_ctc": 16.45,
}

EXPECTED_SVARAH_WER = {
    "tiny": 19.96, "base": 14.53, "small": 10.06, "medium": 7.89, "large": 7.11,
    "large_v3_turbo": 8.10, "parakeet": 11.73, "parakeet_ctc": 15.65, "qwen3": 11.82,
}

EXPECTED_AESRC_WER = {
    "tiny": 13.66, "base": 9.96, "small": 7.23, "medium": 5.73, "large": 5.20,
    "large_v3_turbo": 5.81, "parakeet": 6.26, "parakeet_ctc": 7.50, "qwen3": 5.23,
    "tiny_hf": 17.45, "tiny_aesrc_ft": 12.64,
    "small_hf": 7.22, "small_aesrc_ft": 5.64,
    "medium_hf": 5.63, "medium_aesrc_ft": 4.48,
}

def _check_committed_numbers(dataset, expected):
    import pandas as pd
    summary = os.path.join(stage2_dir(dataset), "wer_summary_all_models.csv")
    if not os.path.exists(summary):
        print(f"[skip] no committed Stage 2 summary; run normalize_and_score.py --dataset {dataset}")
        return
    df = pd.read_csv(summary)
    tc = df[df["mode"] == "transcript_clean"].set_index("model")["corpus_wer_pct"]
    for model, exp in expected.items():
        assert abs(tc[model] - exp) < 0.01, f"{model} drifted: {tc[model]} != {exp}"

def test_committed_tie_numbers_regression():
    _check_committed_numbers("tie", EXPECTED_TIE_WER)

def test_committed_svarah_numbers_regression():
    _check_committed_numbers("svarah", EXPECTED_SVARAH_WER)

def test_committed_aesrc_numbers_regression():
    _check_committed_numbers("aesrc", EXPECTED_AESRC_WER)


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
