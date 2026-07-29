"""
Pre-flight leakage check: speaker overlap across train / validation / test splits.

A dataset's splits are disjoint *sets of clips*, but ASR results are inflated if the same
SPEAKERS appear in both train and test (the model can memorize a speaker's voice). We don't
control the official splits, so the right thing to do is measure and DISCLOSE the overlap.

CPU-only and light: only the speaker column is read (plus, for adapter-loaded datasets,
the adapter's single-clip probe decode). Writes results/<dataset>/analysis/speaker_overlap.md.

Usage:
    python finetune/check_speaker_overlap.py                    # dataset = tie (default)
    python finetune/check_speaker_overlap.py --dataset aesrc
"""

import argparse
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import HF_CACHE, results_dir
from utils.registry import get_dataset

ROLES = ("train", "validation", "eval")


def speaker_ids(dataset_key: str, spec, role: str) -> list[str]:
    if dataset_key == "tie":
        # Direct load, as in the fine-tune scripts: TIE's validation split lacks the
        # duration column the adapter's schema validation requires.
        ds = load_dataset(spec.hf_id, split=spec.splits[role], cache_dir=HF_CACHE)
        return [str(s) for s in ds["Speaker_ID"]]

    from utils.datasets import load_split

    ds, _ = load_split(dataset_key, role)
    return [str(s) for s in ds[spec.speaker_col]]


def main(dataset: str) -> None:
    spec = get_dataset(dataset)
    if not spec.speaker_col and dataset != "tie":
        sys.exit(f"[ERROR] dataset '{dataset}' exposes no speaker column - overlap not computable.")
    roles = [r for r in ROLES if r in spec.splits]
    if {"train", "eval"} - set(roles):
        sys.exit(f"[ERROR] dataset '{dataset}' has no train/eval split pair ({spec.splits}) - "
                 f"nothing to check.")

    print("=" * 70)
    print(f"SPEAKER OVERLAP CHECK (data-leakage pre-flight) — {spec.display}")
    print("=" * 70)

    ids = {role: speaker_ids(dataset, spec, role) for role in roles}
    sets = {role: set(v) for role, v in ids.items()}

    train_set, test_set = sets["train"], sets["eval"]
    val_set = sets.get("validation", set())

    test_in_train = test_set & train_set
    val_in_train = val_set & train_set

    # Share of test *clips* spoken by a speaker that also appears in train.
    test_clips_overlap = sum(1 for sid in ids["eval"] if sid in train_set)
    test_clip_share = test_clips_overlap / len(ids["eval"]) if ids["eval"] else 0.0

    lines = [
        f"# Speaker overlap across splits: {spec.display} (data-leakage disclosure)",
        "",
        "| Split | Clips | Unique speakers |",
        "|-------|------:|----------------:|",
    ]
    for role in roles:
        lines.append(f"| {spec.splits[role]} | {len(ids[role])} | {len(sets[role])} |")

    lines += [
        "",
        "## Train ∩ Test (the relevant leakage)",
        "",
        f"- Test speakers also present in train: **{len(test_in_train)} / {len(test_set)}** "
        f"({len(test_in_train) / max(len(test_set), 1) * 100:.1f}% of test speakers)",
        f"- Test clips spoken by a train-seen speaker: **{test_clips_overlap} / {len(ids['eval'])}** "
        f"({test_clip_share * 100:.1f}% of test clips)",
    ]
    if val_set:
        lines.append(
            f"- Validation speakers also present in train: **{len(val_in_train)} / {len(val_set)}** "
            f"({len(val_in_train) / max(len(val_set), 1) * 100:.1f}% of validation speakers)"
        )
    lines += ["", "## Interpretation", ""]
    if test_clip_share > 0:
        lines.append(
            f"> **Speaker-matched fine-tuning**: {test_clip_share * 100:.1f}% of test clips come from "
            "speakers also seen during training. The fine-tuning improvement therefore partly reflects "
            "speaker adaptation. This is disclosed, not hidden — it reflects the dataset's official splits, "
            "which we did not modify."
        )
    else:
        lines.append(
            "> No speaker overlap between train and test — the fine-tuning gain reflects genuine "
            "generalization to unseen speakers."
        )
    lines.append("")

    print("\n".join(lines))

    out_dir = os.path.join(results_dir(dataset), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "speaker_overlap.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Speaker-overlap leakage check.")
    ap.add_argument("--dataset", default="tie", help="registry dataset key (tie, aesrc, ...)")
    main(ap.parse_args().dataset)
