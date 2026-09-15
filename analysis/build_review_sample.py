"""Build a non-blind human review sheet of clips that several strong models get
wrong at once. One script for all three corpora (TIE, Svarah, AESRC).

The reviewer sees every model's hypothesis and judges, per clip, whether the
high WER comes from the audio, the reference transcript, or the models. Clips
are picked by requiring several of the strongest models to agree a clip is hard,
which filters out model-specific failure modes. Pretrained outputs only, never
the fine-tuned checkpoints: this is about locating hard clips, not about judging
fine-tuning.

Svarah flags far more clips than anyone can review by hand, so it takes a second
step: drop references shorter than MIN_REF_WORDS, where WER can only be 0 or 100
percent, then sample within duration bands with a fixed seed so the reviewed set
spans the corpus instead of piling up in the shortest band.

Review columns, all left empty here for the reviewer to fill:
    reference_check      is the reference transcript itself correct
    corrected_reference  free text, what the clip actually says
    hyp_<model>_check    per-model correctness, since a clip is rarely uniformly
                         right or wrong across all five
    error_type           one cell, comma separated when a clip has more
                         than one cause
    reviewer_decision    the verdict for the row
    reviewer_notes       free text

Writes review_sheet.csv (source of truth) and review_sheet.xlsx (same data with
the review columns constrained to dropdown lists) into the corpus folder. The
derived columns are filled afterwards by analysis/fill_review_checks.py.

Needs the Stage 2 per-clip CSVs, so run `python normalize_and_score.py
--dataset <ds>` first.

Usage:
    .venv/bin/python analysis/build_review_sample.py --dataset tie
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import stage2_dir  # noqa: E402
from analysis.review_common import (  # noqa: E402
    CHECK_OPTIONS, LABELS as ERROR_TYPE_OPTIONS, REVIEW_FOLDERS, REVIEW_MODELS, REVIEWER_DECISION_OPTIONS,
)

HERE = os.path.dirname(os.path.abspath(__file__))

MODE = "transcript_clean"
REQUIRED_MODELS = list(REVIEW_MODELS[:4])
BONUS_MODEL = REVIEW_MODELS[4]
ALL_MODELS = list(REVIEW_MODELS)
WER_THRESHOLD = 40.0  # percent
MIN_MODELS_FLAGGED = 3  # of len(REQUIRED_MODELS)

# Svarah only. Filtering on reference length rather than duration keeps valid
# fast-speech clips and drops two-word ones. Bands are [low, high) in seconds,
# None means no upper bound, and each quota is about 29% of its band.
MIN_REF_WORDS = 3
SAMPLE_PER_BAND = {(0, 2): 8, (2, 4): 16, (4, 6): 11, (6, 9): 11, (9, 14): 10, (14, None): 4}
SAMPLE_SEED = 42

# Per corpus: folder, demographic column from Stage 2, and whether to sample down.
DATASETS = {
    "tie": {"dir": REVIEW_FOLDERS["tie"], "demo": ("native_region", "Native_Region"),
            "ref_words": False, "sample": False},
    "svarah": {"dir": REVIEW_FOLDERS["svarah"], "demo": ("native_language", "Native_Language"),
               "ref_words": True, "sample": True},
    "aesrc": {"dir": REVIEW_FOLDERS["aesrc"], "demo": None,
              # AESRC's only demographic column, Accent, is constant across the subset.
              "ref_words": False, "sample": False},
}



def load_model(dataset: str, model: str) -> dict:
    """Return {sample_id: row_dict} from that model's WER csv."""
    path = os.path.join(stage2_dir(dataset), MODE, f"wer_{model}_{MODE}.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        return {str(row["ID"]): row for row in csv.DictReader(fh)}


def stratified_sample(rows: list) -> list:
    """Pick SAMPLE_PER_BAND clips per duration band from the length-filtered
    pool. Seeded, so the same pool always yields the same sheet. A band holding
    fewer clips than its quota contributes all of them rather than failing."""
    eligible = [r for r in rows if r["ref_words"] >= MIN_REF_WORDS]
    rng = random.Random(SAMPLE_SEED)
    picked = []
    for (lo, hi), quota in SAMPLE_PER_BAND.items():
        band = [
            r for r in eligible
            if r["duration_seconds"] not in ("", None)
            and lo <= float(r["duration_seconds"]) < (hi if hi is not None else float("inf"))
        ]
        take = sorted(band, key=lambda r: r["sample_id"])
        picked.extend(take if len(take) <= quota else rng.sample(take, quota))
        print(f"    {lo:>2}-{str(hi) + 's' if hi else 'max':<5} pool {len(band):>3}  sampled {min(len(band), quota):>3}")
    picked.sort(key=lambda r: (float(r["duration_seconds"]), r["sample_id"]))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    folder = os.path.join(HERE, cfg["dir"])
    csv_path = os.path.join(folder, "review_sheet.csv")
    xlsx_path = os.path.join(folder, "review_sheet.xlsx")

    tables = {m: load_model(args.dataset, m) for m in ALL_MODELS}
    flagged_counts = {m: 0 for m in ALL_MODELS}
    out_rows = []

    for sid in tables[REQUIRED_MODELS[0]]:
        base_row = tables[REQUIRED_MODELS[0]][sid]
        wers, hyps = {}, {}
        for m in ALL_MODELS:
            r = tables[m].get(sid)
            if r is None:
                continue
            wers[m] = float(r["wer"]) * 100
            hyps[m] = r["hypothesis_raw"]
            if wers[m] > WER_THRESHOLD:
                flagged_counts[m] += 1

        n_flagged = sum(1 for m in REQUIRED_MODELS if m in wers and wers[m] > WER_THRESHOLD)
        if n_flagged < MIN_MODELS_FLAGGED:
            continue

        row = {
            "sample_id": sid,
            "audio_filename": f"{sid}.wav",
            "reference": base_row["reference_raw"],
            **{f"hyp_{m}": hyps.get(m, "") for m in ALL_MODELS},
            **{f"wer_{m}": round(wers[m], 2) if m in wers else "" for m in ALL_MODELS},
            "avg_wer": round(sum(wers.values()) / len(wers), 2) if wers else "",
            "n_models_flagged": n_flagged,
            "duration_seconds": base_row.get("Speech_Duration_seconds", ""),
            "reference_check": "",
            "corrected_reference": "",
            **{f"hyp_{m}_check": "" for m in ALL_MODELS},
            "error_type": "",
            "reviewer_decision": "",
            "reviewer_notes": "",
        }
        if cfg["demo"]:
            col, source = cfg["demo"]
            row[col] = base_row.get(source, "")
        if cfg["ref_words"]:
            row["ref_words"] = len(base_row["reference"].split())
        out_rows.append(row)

    middle = ["n_models_flagged"]
    if cfg["demo"]:
        middle.append(cfg["demo"][0])
    middle.append("duration_seconds")
    if cfg["ref_words"]:
        middle.append("ref_words")

    fieldnames = (
        ["sr_no", "sample_id", "audio_filename", "reference"]
        + [f"hyp_{m}" for m in ALL_MODELS]
        + [f"wer_{m}" for m in ALL_MODELS]
        + ["avg_wer"] + middle + ["reference_check", "corrected_reference"]
        + [f"hyp_{m}_check" for m in ALL_MODELS]
        + ["error_type", "reviewer_decision", "reviewer_notes"]
    )

    print(f"[review] threshold: WER > {WER_THRESHOLD:.0f}% on {MODE}, required models: "
          f"{', '.join(REQUIRED_MODELS)} (+{BONUS_MODEL} as a bonus signal, not required)")
    print("[review] per-model flagged clip counts (WER > threshold):")
    for m in ALL_MODELS:
        req = "required" if m in REQUIRED_MODELS else "bonus"
        print(f"    {m:14s} {flagged_counts[m]:4d}  ({req})")
    print(f"[review] flagged pool: {len(out_rows)} clips")

    if cfg["sample"]:
        print(f"[sample] reference >= {MIN_REF_WORDS} words, then stratified by duration:")
        out_rows = stratified_sample(out_rows)
    else:
        out_rows.sort(key=lambda r: r["n_models_flagged"], reverse=True)
    for i, row in enumerate(out_rows, start=1):
        row["sr_no"] = i

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    write_xlsx(fieldnames, out_rows, xlsx_path)
    print(f"[review] wrote {len(out_rows)} rows to {csv_path}")
    print(f"[review] wrote reviewer sheet (with dropdowns) to {xlsx_path}")


def write_xlsx(fieldnames, rows, xlsx_path) -> None:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.comments import Comment

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(fieldnames)

    header_fill = PatternFill("solid", fgColor="DDEBF7")
    check_fill = PatternFill("solid", fgColor="FFF2CC")
    flag_fill = PatternFill("solid", fgColor="E2EFDA")
    decision_fill = PatternFill("solid", fgColor="FCE4D6")

    check_cols = {"reference_check", "corrected_reference",
                  "normalised_corrected_reference"} | {f"hyp_{m}_check" for m in ALL_MODELS}
    decision_cols = {"reviewer_decision", "reviewer_notes"}

    for c, name in enumerate(fieldnames, start=1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        if name in check_cols:
            cell.fill = check_fill
        elif name == "error_type":
            cell.fill = flag_fill
            cell.comment = Comment(
                "Free text. Pick from these, comma-separated if more than one applies:\n"
                + "\n".join(f"- {o}" for o in ERROR_TYPE_OPTIONS),
                "review sheet",
            )
        elif name == "reviewer_decision":
            cell.fill = decision_fill
            # No dropdown: this column sometimes names the model at fault.
            cell.comment = Comment(
                "Free text. Lead with one of these, add a model name after a dash "
                "if one model in particular is at fault:\n"
                + "\n".join(f"- {o}" for o in REVIEWER_DECISION_OPTIONS),
                "review sheet",
            )
        elif name in decision_cols:
            cell.fill = decision_fill
        else:
            cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    for row in rows:
        ws.append([row.get(k, "") for k in fieldnames])
    for r in range(2, len(rows) + 2):
        for c in range(1, len(fieldnames) + 1):
            ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")

    widths = {
        "sr_no": 6, "sample_id": 12, "audio_filename": 16, "reference": 45,
        "avg_wer": 9, "avg_wer_true": 9, "n_models_flagged": 9,
        "native_region": 12, "native_language": 12, "ref_words": 9,
        "duration_seconds": 10, "reference_check": 16, "corrected_reference": 40,
        "normalised_corrected_reference": 40, "error_type": 30,
        "reviewer_decision": 20, "reviewer_notes": 30, "audio_path": 30,
    }
    for c, name in enumerate(fieldnames, start=1):
        if name in widths:
            width = widths[name]
        elif name.startswith("hyp_") and name.endswith("_check"):
            width = 14
        elif name.startswith("hyp_"):
            width = 35
        elif name.startswith("wer_"):
            width = 8
        else:
            width = 12
        ws.column_dimensions[get_column_letter(c)].width = width
    ws.freeze_panes = "E2"

    def add_dropdown(colname, options):
        if colname not in fieldnames:
            return
        letter = get_column_letter(fieldnames.index(colname) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"',
                            allow_blank=True, showDropDown=False)
        dv.error = "Pick one of the listed options."
        dv.errorTitle = "Invalid entry"
        ws.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{len(rows) + 1}")

    add_dropdown("reference_check", CHECK_OPTIONS)
    for m in ALL_MODELS:
        add_dropdown(f"hyp_{m}_check", CHECK_OPTIONS)

    wb.save(xlsx_path)


if __name__ == "__main__":
    main()
