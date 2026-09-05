"""Fail-closed guards for paired statistical comparisons."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import statistics
from analysis.compare_all import _validate_headline_panels
from analysis.compare_finetune import paired_speaker_bootstrap
from utils import inference_loop


def _table(reference: str, errors: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "errors": [errors],
        "ref_words": [len(reference.split())],
        "reference": [reference],
        "speaker": ["speaker-1"],
    }, index=pd.Index(["clip-1"], name="ID"))


def test_paired_statistics_reject_nonpositive_bootstrap_count():
    with pytest.raises(ValueError, match="positive"):
        statistics.analyze("tie", "transcript_clean", B=0)


def test_paired_statistics_reject_reference_mismatch(monkeypatch):
    tables = {
        "tiny": _table("the same reference"),
        "base": _table("a different reference"),
    }
    monkeypatch.setattr(statistics, "models_for_dataset", lambda dataset: ("tiny", "base"))
    monkeypatch.setattr(
        statistics,
        "_load_clip_table",
        lambda dataset, model, mode: tables[model],
    )
    with pytest.raises(ValueError, match="normalized references differ"):
        statistics.analyze("tie", "transcript_clean", B=10)


def test_finetune_bootstrap_rejects_different_panels():
    base = pd.DataFrame({
        "ID": ["clip-1"], "reference": ["hello"], "hypothesis": ["hello"],
        "Speaker_ID": ["speaker-1"],
    })
    fine_tuned = pd.DataFrame({
        "ID": ["clip-2"], "reference": ["hello"], "hypothesis": ["hello"],
    })
    with pytest.raises(ValueError, match="no common clip IDs"):
        paired_speaker_bootstrap(base, fine_tuned, B=10)


def test_headline_summary_rejects_different_panels():
    tiny = pd.DataFrame({
        "ID": ["clip-1"], "reference": ["hello"], "hypothesis": ["hello"],
    })
    base = pd.DataFrame({
        "ID": ["clip-2"], "reference": ["hello"], "hypothesis": ["hello"],
    })
    with pytest.raises(ValueError, match="headline evaluation panel differs"):
        _validate_headline_panels(
            {("tiny", "transcript_clean"): tiny,
             ("base", "transcript_clean"): base},
            "transcript_clean",
        )


def test_sigterm_handler_writes_the_checkpoint_resume_path(monkeypatch):
    captured = {}
    saved = {}

    monkeypatch.setattr(
        inference_loop.signal,
        "signal",
        lambda sig, handler: captured.setdefault("handler", handler),
    )
    monkeypatch.setattr(
        inference_loop,
        "save_checkpoint",
        lambda rows, model, dataset: saved.update(
            rows=rows, model=model, dataset=dataset
        ) or "/tmp/wer_tiny_partial.csv",
    )
    inference_loop._install_sigterm_handler({
        "rows": [{"ID": "clip-1"}], "model": "tiny", "dataset": "tie",
    })

    with pytest.raises(SystemExit) as exc:
        captured["handler"](None, None)
    assert exc.value.code == 143
    assert saved == {
        "rows": [{"ID": "clip-1"}], "model": "tiny", "dataset": "tie",
    }
