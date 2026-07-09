"""
Pre-flight leakage check: speaker overlap across train / validation / test splits.

The dataset's splits are disjoint *sets of clips*, but ASR results are inflated if the same
SPEAKERS appear in both train and test (the model can memorize a speaker's voice). We don't
control the official splits, so the right thing to do is measure and DISCLOSE the overlap.

Reads only the Speaker_ID column (audio is never decoded), so this is fast and CPU-only.
Writes results/tie/analysis/speaker_overlap.md.

Usage:
    python finetune/check_speaker_overlap.py
"""

import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import HF_CACHE, results_dir

SPLITS = ("train", "validation", "test")


def speaker_ids(split: str) -> list[str]:
    ds = load_dataset("raianand/TIE_shorts", split=split, cache_dir=HF_CACHE)
    # Accessing a single column does not trigger audio decoding.
    return [str(s) for s in ds["Speaker_ID"]]


print("=" * 70)
print("SPEAKER OVERLAP CHECK (data-leakage pre-flight)")
print("=" * 70)

ids = {split: speaker_ids(split) for split in SPLITS}
sets = {split: set(v) for split, v in ids.items()}

train_set, test_set, val_set = sets["train"], sets["test"], sets["validation"]

test_in_train = test_set & train_set
val_in_train = val_set & train_set

# Share of test *clips* spoken by a speaker that also appears in train.
test_clips_overlap = sum(1 for sid in ids["test"] if sid in train_set)
test_clip_share = test_clips_overlap / len(ids["test"]) if ids["test"] else 0.0

lines = [
    "# Speaker Overlap Across Splits (data-leakage disclosure)",
    "",
    "| Split | Clips | Unique speakers |",
    "|-------|------:|----------------:|",
]
for split in SPLITS:
    lines.append(f"| {split} | {len(ids[split])} | {len(sets[split])} |")

lines += [
    "",
    "## Train ∩ Test (the relevant leakage)",
    "",
    f"- Test speakers also present in train: **{len(test_in_train)} / {len(test_set)}** "
    f"({len(test_in_train) / max(len(test_set), 1) * 100:.1f}% of test speakers)",
    f"- Test clips spoken by a train-seen speaker: **{test_clips_overlap} / {len(ids['test'])}** "
    f"({test_clip_share * 100:.1f}% of test clips)",
    f"- Validation speakers also present in train: **{len(val_in_train)} / {len(val_set)}** "
    f"({len(val_in_train) / max(len(val_set), 1) * 100:.1f}% of validation speakers)",
    "",
    "## Interpretation",
    "",
]
if test_clip_share > 0:
    lines.append(
        f"> ⚠️ **Speaker-matched fine-tuning**: {test_clip_share * 100:.1f}% of test clips come from "
        "speakers also seen during training. The fine-tuning improvement therefore partly reflects "
        "speaker adaptation. This is disclosed, not hidden — it reflects the dataset's official splits, "
        "which we did not modify."
    )
else:
    lines.append(
        "> ✅ No speaker overlap between train and test — the fine-tuning gain reflects genuine "
        "generalization to unseen speakers."
    )
lines.append("")

print("\n".join(lines))

out_dir = os.path.join(results_dir(), "analysis")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "speaker_overlap.md")
with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {out_path}")
