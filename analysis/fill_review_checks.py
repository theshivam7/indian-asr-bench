"""Fill a review sheet's derived columns from the reviewer's corrected_reference.

One script for all three human review sheets (TIE, Svarah, AESRC), which share
the protocol: the reviewer listens to each clip and types corrected_reference,
and everything else below is derived from it.

  1. normalised_corrected_reference: corrected_reference through a math-notation
     pass (Greek letters, sub/superscripts, operators) and then the project's
     transcript_clean normalizer, so it is comparable to how hypotheses are
     normalized everywhere else in the repo.
  2. wer_<model>_true and avg_wer_true: WER against the corrected reference
     instead of the dataset reference.
  3. reference_check and hyp_<model>_check, by WER thresholds
     (<=8% Correct, <=40% Partially correct, >40% Incorrect).
  4. error_type, reviewer_decision and reviewer_notes, but only what a text
     diff supports: misalignment, reference error, long numbers, repeated
     phrases, very short clips. Anything that needs a human ear comes from the
     folder's error_types.csv instead.

Three sources feed the reviewer columns, in this order of precedence:

  error_types.csv  the final per-clip reading of what went wrong, written after
                   going through every row one by one. Carries error_type,
                   reviewer_notes and, on a few clips, reviewer_decision. Its
                   workbook_error_type column keeps whatever label the reviewer
                   typed on that clip, where it differs from the final one.
  the sheet        anything already in the review columns of the input sheet.
  the rules above  used only where neither of the other two says anything.

Every place a later source overrides an earlier one is listed at the end of the
report, so nothing is lost silently.

Usage:
    .venv/bin/python analysis/fill_review_checks.py --dataset tie
    .venv/bin/python analysis/fill_review_checks.py --dataset svarah
    .venv/bin/python analysis/fill_review_checks.py --dataset aesrc
"""

import argparse
import csv
import datetime
import os
import re
import string
import sys
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.normalize import normalize_text  # noqa: E402
from utils.wer_compute import reference_word_recall  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ALL_MODELS = ["large", "parakeet", "parakeet_ctc", "qwen3", "medium"]

# Per corpus: the folder and the columns between avg_wer_true and reference_check.
DATASETS = {
    "tie": {"dir": "tie_validation", "sheet": "review_sheet",
            "middle": ["n_models_flagged", "native_region", "duration_seconds"]},
    "svarah": {"dir": "svarah_validation", "sheet": "review_sheet",
               "middle": ["n_models_flagged", "native_language",
                          "duration_seconds", "ref_words"]},
    "aesrc": {"dir": "aesrc_validation", "sheet": "review_sheet",
              "middle": ["n_models_flagged", "duration_seconds"]},
}

# One vocabulary across all three sheets. Comma separated when a clip has several causes.
LABELS = [
    "Reference error", "Misalignment", "Truncated audio", "Disfluency",
    "Number formatting", "Technical vocabulary", "Acronym or code",
    "Hindi named entity", "Indian-language named entity",
    "Foreign named entity", "English name or rare word",
    "Brand or product name", "Accent / pronunciation", "Spelling variant",
    "Short utterance",
]

# Both conditions must hold for a clip to count as short.
SHORT_WORDS = 4
SHORT_SECONDS = 3.0

# Columns the reviewer fills in. Everything else comes from the sheet as built,
# so a spreadsheet round trip cannot reformat a WER or drop a hypothesis.
REVIEW_COLS = [
    "reference_check", "corrected_reference",
    *[f"hyp_{m}_check" for m in ALL_MODELS],
    "error_type", "reviewer_decision", "reviewer_notes",
]


CORRECT_MAX_WER = 0.08
PARTIAL_MAX_WER = 0.40

# Below this the reference shares too few words with the true transcript to be a
# noisy version of the same clip. Taken from the gap in the TIE distribution:
# topic-mismatch rows sit at 0.10-0.32 recall, everything else at 0.42 or above.
MISALIGNMENT_RECALL_MAX = 0.35

LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")

SUBSCRIPT_MAP = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ᵢ": "i", "ⱼ": "j", "ₖ": "k", "ₙ": "n",
    "ᵣ": "r", "ₚ": "p", "ᵥ": "v",
}
GREEK_MAP = {
    "π": "pi", "ρ": "rho", "θ": "theta", "ξ": "xi",
    "μ": "mu", "ε": "epsilon", "β": "beta", "Γ": "gamma",
    "ω": "omega",
}
MATH_NOTATION_CHARS = (
    set(SUBSCRIPT_MAP) | set(GREEK_MAP) | set("²³¹ⁿ⁺⁻′×−≠_") | {"̂", "̃"}
)


def find_long_numbers(text: str) -> list[str]:
    """IDs/codes spoken digit-by-digit (e.g. an IC part number "74138"), as
    opposed to a genuine quantity. The normalizer spells any number out as a
    single cardinal ("seventy-four thousand..."), which won't match a model
    that (correctly) transcribed it digit by digit, and can inflate WER here
    without that being a real hypothesis error.
    """
    return LONG_NUMBER_RE.findall(text)


def find_repeated_phrase(text: str) -> str | None:
    """An adjacent repeated 2-4 word run in the human-corrected transcript,
    a text signature of genuine spoken disfluency (stammer/restart) rather
    than a transcription mistake, since it survived manual correction.
    """
    words = [w.strip(string.punctuation).lower() for w in text.split()]
    words = [w for w in words if w]
    for n in (4, 3, 2):
        for i in range(len(words) - 2 * n + 1):
            if words[i:i + n] == words[i + n:i + 2 * n]:
                return " ".join(words[i:i + n])
    return None


def convert_math_notation(text: str) -> str:
    """Rewrite Greek letters, sub/superscripts, and math operators as spoken
    English, so a WER comparison against ASR hypotheses (which only ever
    output spoken words) isn't penalized for symbolic notation alone.

    Mapping is grounded in the exact symbol set actually used in this sheet's
    corrected_reference column (checked once via a full character scan), not
    a generic math-to-text library.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)

    text = text.replace("_", " ")  # subscript join, e.g. I_bias -> I bias

    text = re.sub(r"(\w)̂", r"\1 hat ", text)   # combining circumflex, beta-hat
    text = re.sub(r"(\w)̃", r"\1 tilde ", text)  # combining tilde, u-tilde
    text = text.replace("ẍ", " x double dot ")   # composed x-with-diaeresis (physics ddot notation)

    text = text.replace("⁻¹", " inverse ")  # superscript minus-one, A^-1

    text = text.replace("⁺", " plus ")   # superscript plus
    text = text.replace("⁻", " minus ")  # superscript minus

    text = text.replace("²", " squared ")
    text = text.replace("³", " cubed ")
    text = text.replace("¹", " to the power one ")
    text = text.replace("ⁿ", " to the power n ")

    for ch, word in SUBSCRIPT_MAP.items():
        text = text.replace(ch, " " + word + " ")

    text = text.replace("′", " prime")  # derivative prime, u'

    for ch, word in GREEK_MAP.items():
        text = text.replace(ch, " " + word + " ")

    text = text.replace("×", " into ")    # multiplication sign
    text = text.replace("−", " minus ")   # unicode minus sign
    text = re.sub(r"(?<=\s)-(?=\s)", " minus ", text)  # ascii hyphen used as subtraction (spaced)
    text = text.replace("=", " equal to ")
    text = text.replace("/", " by ")
    text = text.replace("%", " percent ")
    text = text.replace("+", " plus ")
    text = text.replace("*", " star ")
    text = text.replace("≠", " not equal to ")

    return text


def normalize_for_compare(text: str) -> str:
    return normalize_text(convert_math_notation(text or ""))


def word_wer(ref: str, hyp: str):
    import jiwer
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0 if hyp else 0.0
    if not hyp:
        return 1.0
    return jiwer.wer(ref, hyp)


def classify(wer: float) -> str:
    if wer <= CORRECT_MAX_WER:
        return "Correct"
    if wer <= PARTIAL_MAX_WER:
        return "Partially correct"
    return "Incorrect"


coerced: list[str] = []


def cell_text(value) -> str:
    """Undo the two ways a spreadsheet retypes a transcript the reviewer meant
    as text: a bare number 1109 comes back as 1109.0, where the normalizer then
    spells out a trailing "point zero", and a date like 24 July 1997 comes back
    as a datetime. Both are put back the way they were typed, and the caller
    prints which cells this touched."""
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        coerced.append(f"{value.day} {value:%B %Y}")
        return coerced[-1]
    if isinstance(value, float) and value.is_integer():
        coerced.append(str(int(value)))
        return coerced[-1]
    return str(value).strip()


def read_review(path: str) -> dict[str, dict]:
    """sample_id -> the reviewer's cells, as text."""
    if path.lower().endswith(".xlsx"):
        from openpyxl import load_workbook
        rows = list(load_workbook(path, data_only=True).active.values)
        header = [str(h) for h in rows[0]]
        records = [dict(zip(header, r)) for r in rows[1:]]
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            records = list(csv.DictReader(fh))
    out = {}
    for rec in records:
        out[str(rec["sample_id"])] = {c: cell_text(rec.get(c)) for c in REVIEW_COLS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--in", dest="in_path",
                    help="sheet to read the reviewer columns from (default: the sheet itself)")
    ap.add_argument("--base", help="source of every non-reviewer column (default: the sheet)")
    ap.add_argument("--out-csv")
    ap.add_argument("--out-xlsx")
    ap.add_argument("--error-types",
                    help="reviewed per-clip error_type, notes and decisions; wins over the rest")
    ap.add_argument("--report")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    folder = os.path.join(HERE, cfg["dir"])
    sheet_csv = os.path.join(folder, cfg["sheet"] + ".csv")
    args.in_path = args.in_path or sheet_csv
    args.base = args.base or sheet_csv
    args.out_csv = args.out_csv or sheet_csv
    args.out_xlsx = args.out_xlsx or os.path.join(folder, cfg["sheet"] + ".xlsx")
    args.error_types = args.error_types or os.path.join(folder, "error_types.csv")
    args.report = args.report or os.path.join(folder, "review_report.txt")

    review = read_review(args.in_path)
    if coerced:
        print(f"[fill] the spreadsheet had retyped {len(coerced)} reviewer cells as "
              f"numbers or dates, read back as text: {', '.join(coerced)}")
    with open(args.base, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    has_audio_path = "audio_path" in rows[0]

    missing = [r["sample_id"] for r in rows if r["sample_id"] not in review]
    if missing:
        raise SystemExit(f"{len(missing)} clips are not in {args.in_path}: {missing[:3]}")
    for row in rows:
        row.update(review[row["sample_id"]])

    final, typed = {}, []
    if os.path.exists(args.error_types):
        with open(args.error_types, newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                final[rec["sample_id"]] = {
                    c: (rec.get(c) or "").strip()
                    for c in ("error_type", "reviewer_decision", "reviewer_notes")
                }
                was = (rec.get("workbook_error_type") or "").strip()
                if was and was != (rec.get("error_type") or "").strip():
                    typed.append((rec["sr_no"], was, rec["error_type"]))

    fieldnames = [
        "sr_no", "sample_id", "audio_filename", "reference",
        *[f"hyp_{m}" for m in ALL_MODELS],
        *[f"wer_{m}" for m in ALL_MODELS],
        "avg_wer",
        *[f"wer_{m}_true" for m in ALL_MODELS],
        "avg_wer_true",
        *cfg["middle"],
        "reference_check", "corrected_reference", "normalised_corrected_reference",
        *[f"hyp_{m}_check" for m in ALL_MODELS],
        "error_type", "reviewer_decision", "reviewer_notes",
    ]
    if has_audio_path:
        fieldnames.append("audio_path")

    report = []
    disagreements = []

    for row in rows:
        reviewed = final.get(row["sample_id"], {})

        def keep(col, auto):
            """Apply the precedence order and log anything it overrides."""
            human = (row.get(col) or "").strip()
            if human.lower() == "none":
                human = ""
            chosen = reviewed.get(col, "")
            if chosen:
                beaten = human or auto
                if beaten and beaten != chosen:
                    disagreements.append((row["sr_no"], col, "error_types.csv", chosen, beaten))
                return chosen
            if human:
                if auto and human != auto:
                    disagreements.append((row["sr_no"], col, "the sheet", human, auto))
                return human
            return auto

        ncr_raw = row["corrected_reference"]
        ncr = normalize_for_compare(ncr_raw)
        row["normalised_corrected_reference"] = ncr

        report.append(f"=== row {row['sr_no']}  ({row['sample_id']}) ===")
        report.append(f"normalised_corrected_reference: {ncr}")

        ref_norm = normalize_for_compare(row["reference"])
        ref_wer = word_wer(ncr, ref_norm)
        row["reference_check"] = keep("reference_check", classify(ref_wer))
        report.append(f"reference_check   = {row['reference_check']:18s} (wer={ref_wer:.2f})")
        report.append(f"  reference (norm): {ref_norm}")

        hyp_checks, wers = {}, []
        for m in ALL_MODELS:
            hyp_norm = normalize_for_compare(row[f"hyp_{m}"])
            hyp_wer = word_wer(ncr, hyp_norm)
            wers.append(round(hyp_wer * 100, 2))
            row[f"wer_{m}_true"] = f"{hyp_wer * 100:.2f}"
            check = keep(f"hyp_{m}_check", classify(hyp_wer))
            hyp_checks[m] = check
            row[f"hyp_{m}_check"] = check
            report.append(f"hyp_{m:13s}_check = {check:18s} (wer_true={hyp_wer * 100:.2f}, was {row[f'wer_{m}']})")
            report.append(f"  hyp_{m} (norm): {hyp_norm}")
        row["avg_wer_true"] = f"{sum(wers) / len(wers):.2f}"

        n_ok = sum(1 for c in hyp_checks.values() if c in ("Correct", "Partially correct"))
        n_incorrect = sum(1 for c in hyp_checks.values() if c == "Incorrect")

        ref_recall = reference_word_recall(ncr, ref_norm)
        is_misaligned = ref_recall < MISALIGNMENT_RECALL_MAX

        error_types, notes = [], []

        if is_misaligned:
            error_types.append("Misalignment")
            notes.append(
                f"The reference shares only {ref_recall:.0%} of the true transcript's words, "
                "so it looks like a different clip rather than a noisy transcription."
            )
        elif row["reference_check"] != "Correct":
            error_types.append("Reference error")

        long_nums = find_long_numbers(ncr_raw)
        if long_nums:
            error_types.append("Number formatting")
            notes.append(
                f"Holds a long number ({', '.join(long_nums)}) the speaker reads out digit "
                "by digit, which the normalizer spells as one cardinal."
            )
        elif any(ch in MATH_NOTATION_CHARS for ch in ncr_raw):
            error_types.append("Technical vocabulary")

        repeat = find_repeated_phrase(ncr_raw)
        if repeat:
            error_types.append("Disfluency")
            notes.append(f'The speaker repeats "{repeat}"; the reference drops it.')

        words = len(ncr_raw.split())
        try:
            seconds = float(row.get("duration_seconds") or 0)
        except ValueError:
            seconds = 0.0
        if words <= SHORT_WORDS and 0 < seconds < SHORT_SECONDS:
            error_types.append("Short utterance")

        row["error_type"] = keep("error_type", ", ".join(dict.fromkeys(error_types)))
        row["reviewer_notes"] = keep("reviewer_notes", " ".join(notes))

        decision = ""
        if is_misaligned:
            decision = "Reference error"
        elif row["reference_check"] != "Correct" and n_ok >= 3:
            decision = "Reference error"
        elif row["reference_check"] in ("Correct", "Partially correct") and n_incorrect >= 3:
            decision = "Genuine model error"
        row["reviewer_decision"] = keep("reviewer_decision", decision)

        report.append(f"ref_recall = {ref_recall:.2f}   error_type -> {row['error_type'] or '(none)'}")
        report.append(f"reviewer_decision -> {row['reviewer_decision'] or '(left blank, ambiguous)'}")
        if row["reviewer_notes"]:
            report.append(f"reviewer_notes -> {row['reviewer_notes']}")
        report.append("")

    if typed:
        report.append("=== error_type the reviewer typed, and the final label ===")
        for sr, was, now in typed:
            report.append(f"row {sr}: reviewer typed \"{was}\", final label is \"{now}\"")
        report.append("")

    if disagreements:
        report.append("=== cells where one source overrode another ===")
        for sr, col, src, chosen, beaten in disagreements:
            report.append(f"row {sr}  {col}: {src} says \"{chosen}\", overriding \"{beaten}\"")
        report.append("")

    out_rows = [{k: row.get(k, "") for k in fieldnames} for row in rows]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[fill] wrote {len(out_rows)} rows to {args.out_csv}")

    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    print(f"[fill] wrote diff report to {args.report}")

    write_xlsx(fieldnames, out_rows, args.out_xlsx)
    print(f"[fill] wrote reviewer sheet (with dropdowns) to {args.out_xlsx}")
    print(f"[fill] cells where one source overrode another: {len(disagreements)}")


def write_xlsx(fieldnames, rows, xlsx_path) -> None:
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.comments import Comment

    REFERENCE_CHECK_OPTIONS = ["Correct", "Partially correct", "Incorrect"]
    HYP_CHECK_OPTIONS = ["Correct", "Partially correct", "Incorrect"]
    REVIEWER_DECISION_OPTIONS = [
        "Genuine model error", "Reference error", "Audio artifact",
        "Not a real error", "Unsure",
    ]
    ERROR_TYPE_OPTIONS = [
        "Noise / audio quality", "Speed (fast or slow / unclear)",
        "Accent / pronunciation", "Technical vocabulary (jargon, numbers, names)",
        "Disfluency (fillers, repetitions, false starts)",
        "Code-switching (non-English words)", "Reference error",
        "Misalignment (wrong clip boundary)", "Other",
    ]

    wb = Workbook()
    ws = wb.active
    ws.title = "review"

    ws.append(fieldnames)
    header_fill = PatternFill("solid", fgColor="DDEBF7")
    check_fill = PatternFill("solid", fgColor="FFF2CC")
    flag_fill = PatternFill("solid", fgColor="E2EFDA")
    decision_fill = PatternFill("solid", fgColor="FCE4D6")

    check_cols = {"reference_check", "corrected_reference", "normalised_corrected_reference"} | {
        f"hyp_{m}_check" for m in ALL_MODELS
    }
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
            # No dropdown: this column sometimes names the model at fault, which a
            # closed list cannot hold. A header comment stands in for validation.
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
        ws.append([row[k] for k in fieldnames])

    for r in range(2, len(rows) + 2):
        for c in range(1, len(fieldnames) + 1):
            ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")

    widths = {
        "sr_no": 6, "sample_id": 12, "audio_filename": 16, "reference": 45,
        "avg_wer": 9, "avg_wer_true": 9, "n_models_flagged": 9, "native_region": 12,
        "duration_seconds": 10, "reference_check": 16, "corrected_reference": 40,
        "normalised_corrected_reference": 40,
        "error_type": 30, "reviewer_decision": 20, "reviewer_notes": 30,
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
        c = fieldnames.index(colname) + 1
        letter = get_column_letter(c)
        formula = '"' + ",".join(options) + '"'
        dv = DataValidation(type="list", formula1=formula, allow_blank=True, showDropDown=False)
        dv.error = "Pick one of the listed options."
        dv.errorTitle = "Invalid entry"
        ws.add_data_validation(dv)
        dv.add(f"{letter}2:{letter}{len(rows) + 1}")

    add_dropdown("reference_check", REFERENCE_CHECK_OPTIONS)
    for m in ALL_MODELS:
        add_dropdown(f"hyp_{m}_check", HYP_CHECK_OPTIONS)
    # reviewer_decision has no dropdown, see its header comment above.

    wb.save(xlsx_path)




if __name__ == "__main__":
    main()
