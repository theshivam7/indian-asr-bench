"""
Cross-model, cross-mode WER comparison and visualization.

Reads result CSVs from results/stage2_processed/{mode}/
Produces summary tables, breakdowns, and charts.

Run after normalize_and_score.py has completed for all models.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.wer_compute import compute_corpus_wer
from utils.normalize import MODES
from utils.io_helpers import build_md_table

MODELS = ("base", "medium", "large", "parakeet", "qwen3", "medium_hf", "medium_ft")
PRIMARY_MODE = "transcript_clean"  # gold standard mode for breakdowns and charts

MODEL_DISPLAY = {
    "base": "Whisper Base",
    "medium": "Whisper Medium",
    "large": "Whisper Large",
    "parakeet": "Parakeet-TDT-0.6B",
    "qwen3": "Qwen3-ASR-1.7B",
    "medium_hf": "Whisper Medium (HF)",
    "medium_ft": "Whisper Medium (FT)",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
STAGE2_DIR = os.path.join(RESULTS_DIR, "stage2_processed")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)



def load_result_csv(model: str, mode: str) -> pd.DataFrame | None:
    path = os.path.join(STAGE2_DIR, mode, f"wer_{model}_{mode}.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    return pd.read_csv(path)


# --------------- 1. Build corpus WER matrix ---------------
print("=" * 70)
print("BUILDING WER SUMMARY MATRIX")
print("=" * 70)

summary_rows = []
all_data: dict[tuple[str, str], pd.DataFrame] = {}

for model in MODELS:
    row = {"model": model}
    for mode in MODES:
        df = load_result_csv(model, mode)
        if df is None:
            row[mode] = None
            continue
        all_data[(model, mode)] = df
        refs = df["reference"].fillna("").tolist()
        hyps = df["hypothesis"].fillna("").tolist()
        stats = compute_corpus_wer(refs, hyps)
        row[mode] = round(stats["corpus_wer"] * 100, 2)
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(ANALYSIS_DIR, "wer_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"\nSummary matrix saved to: {summary_csv}")
print(df_summary.to_string(index=False))

# --------------- 2. Breakdowns (using standard_num as primary) ---------------
breakdown_configs = [
    ("Native_Region", "comparison_by_region.csv"),
    ("Speech_Class", "comparison_by_speech_class.csv"),
    ("Gender", "comparison_by_gender.csv"),
    ("Discipline_Group", "comparison_by_discipline.csv"),
]

print(f"\n{'=' * 70}")
print(f"BREAKDOWNS (mode: {PRIMARY_MODE})")
print("=" * 70)

for col, filename in breakdown_configs:
    rows = []
    for model in MODELS:
        key = (model, PRIMARY_MODE)
        if key not in all_data:
            continue
        df = all_data[key]
        if col not in df.columns:
            continue
        for group_val, group_df in df.groupby(col):
            refs = group_df["reference"].fillna("").tolist()
            hyps = group_df["hypothesis"].fillna("").tolist()
            stats = compute_corpus_wer(refs, hyps)
            rows.append({
                "model": model,
                col: group_val,
                "corpus_wer_pct": round(stats["corpus_wer"] * 100, 2),
                "num_samples": stats["num_samples"],
            })

    if rows:
        df_bd = pd.DataFrame(rows)
        out_path = os.path.join(ANALYSIS_DIR, filename)
        df_bd.to_csv(out_path, index=False)
        print(f"\n  {col} breakdown saved to: {out_path}")
        print(df_bd.to_string(index=False))

# --------------- 3. Charts ---------------
print(f"\n{'=' * 70}")
print("GENERATING CHARTS")
print("=" * 70)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
# Keep each model-group within ~0.8 of a unit so adjacent groups don't overlap,
# regardless of how many models are present (5 pretrained + medium_hf + medium_ft).
bar_width = 0.8 / len(MODELS)

# Chart 0 (headline): models ranked by WER on the gold mode — clearest single view.
gold_rows = []
for model in MODELS:
    row = df_summary[df_summary["model"] == model]
    if row.empty or pd.isna(row[PRIMARY_MODE].values[0]):
        continue
    gold_rows.append((MODEL_DISPLAY.get(model, model), float(row[PRIMARY_MODE].values[0])))

if gold_rows:
    gold_rows.sort(key=lambda t: t[1], reverse=True)  # worst on top, best at bottom
    names = [n for n, _ in gold_rows]
    vals = [v for _, v in gold_rows]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.5))
    bars = ax.barh(names, vals, color="#4878a8")
    bars[-1].set_color("#2ca02c")  # best (lowest WER) highlighted green
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Corpus WER (%)")
    ax.set_title(f"Model ranking by WER  ({PRIMARY_MODE}, lower is better)")
    ax.margins(x=0.12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    rank_path = os.path.join(ANALYSIS_DIR, "wer_by_model.png")
    fig.savefig(rank_path)
    plt.close(fig)
    print(f"  Saved: {rank_path}")

# Chart 1: WER by model and mode (grouped bar)
fig, ax = plt.subplots(figsize=(10, 6))
x_labels = list(MODES)
x = range(len(x_labels))

for i, model in enumerate(MODELS):
    values = []
    for mode in MODES:
        row = df_summary[df_summary["model"] == model]
        val = row[mode].values[0] if not row.empty and pd.notna(row[mode].values[0]) else 0
        values.append(val)
    offset = (i - (len(MODELS) - 1) / 2) * bar_width
    bars = ax.bar([xi + offset for xi in x], values, bar_width, label=MODEL_DISPLAY.get(model, model))
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Evaluation Mode")
ax.set_ylabel("WER (%)")
ax.set_title("WER by Model and Evaluation Mode")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
chart_path = os.path.join(ANALYSIS_DIR, "wer_by_model_and_mode.png")
fig.savefig(chart_path)
plt.close(fig)
print(f"  Saved: {chart_path}")

# Chart 2: per-utterance WER distribution, one panel per model (small multiples).
# Y = % of that model's utterances, X = per-sample WER in 5% bins (0-100%).
# WER > 100% (insertion-heavy) is clipped into the final bin for display.
models_with_data = [m for m in MODELS if (m, PRIMARY_MODE) in all_data]
if models_with_data:
    ncols = 2
    nrows = math.ceil(len(models_with_data) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows + 1),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    dist_bins = [i * 5 for i in range(21)]  # 0, 5, ..., 100
    for ax_i, model_name in enumerate(models_with_data):
        ax = axes[ax_i]
        wer_pct = (all_data[(model_name, PRIMARY_MODE)]["wer"].dropna().clip(upper=1.0) * 100).values
        if len(wer_pct):
            weights = [100.0 / len(wer_pct)] * len(wer_pct)
            ax.hist(wer_pct, bins=dist_bins, weights=weights, color="#4878a8", edgecolor="white")
            median = float(pd.Series(wer_pct).median())
            ax.axvline(median, color="#d62728", linestyle="--", linewidth=1)
            ax.text(median + 2, ax.get_ylim()[1] * 0.85, f"median {median:.0f}%",
                    fontsize=8, color="#d62728")
        ax.set_title(MODEL_DISPLAY.get(model_name, model_name), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(models_with_data), len(axes)):
        axes[j].axis("off")
    fig.supxlabel("WER (%) — bin width 5%")
    fig.supylabel("% of utterances")
    fig.suptitle(f"Per-utterance WER distribution ({PRIMARY_MODE})", fontsize=12)
    fig.tight_layout()
    dist_path = os.path.join(ANALYSIS_DIR, "wer_distribution.png")
    fig.savefig(dist_path)
    plt.close(fig)
    print(f"  Saved: {dist_path}")

# Chart 3: WER by duration bucket (standard_num mode)
duration_data = {}
for model_name in MODELS:
    key = (model_name, PRIMARY_MODE)
    if key not in all_data:
        continue
    df = all_data[key].copy()
    df["Speech_Duration_seconds"] = pd.to_numeric(df["Speech_Duration_seconds"], errors="coerce")
    df["duration_bucket"] = pd.cut(
        df["Speech_Duration_seconds"],
        bins=[0, 5, 15, 30, 60, float("inf")],
        labels=["0-5s", "5-15s", "15-30s", "30-60s", "60s+"],
    )
    for bucket, group_df in df.groupby("duration_bucket", observed=True):
        refs = group_df["reference"].fillna("").tolist()
        hyps = group_df["hypothesis"].fillna("").tolist()
        stats = compute_corpus_wer(refs, hyps)
        duration_data.setdefault(str(bucket), {})[model_name] = round(stats["corpus_wer"] * 100, 2)

if duration_data:
    fig, ax = plt.subplots(figsize=(10, 6))
    buckets = ["0-5s", "5-15s", "15-30s", "30-60s", "60s+"]
    buckets = [b for b in buckets if b in duration_data]
    x = range(len(buckets))
    for i, model_name in enumerate(MODELS):
        values = [duration_data.get(b, {}).get(model_name, 0) for b in buckets]
        offset = (i - (len(MODELS) - 1) / 2) * bar_width
        ax.bar([xi + offset for xi in x], values, bar_width, label=MODEL_DISPLAY.get(model_name, model_name))
    ax.set_xlabel("Duration Bucket")
    ax.set_ylabel("WER (%)")
    ax.set_title(f"WER by Duration (mode: {PRIMARY_MODE})")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    dur_chart = os.path.join(ANALYSIS_DIR, "wer_by_duration.png")
    fig.savefig(dur_chart)
    plt.close(fig)
    print(f"  Saved: {dur_chart}")

    dur_rows = []
    for bucket in buckets:
        for model_name in MODELS:
            if model_name in duration_data.get(bucket, {}):
                dur_rows.append({
                    "duration_bucket": bucket,
                    "model": model_name,
                    "corpus_wer_pct": duration_data[bucket][model_name],
                })
    if dur_rows:
        dur_csv = os.path.join(ANALYSIS_DIR, "comparison_by_duration.csv")
        pd.DataFrame(dur_rows).to_csv(dur_csv, index=False)
        print(f"  Saved: {dur_csv}")

# Chart 4: WER by region (standard_num mode)
for col, chart_name in [("Native_Region", "wer_by_region.png")]:
    region_data = {}
    for model in MODELS:
        key = (model, PRIMARY_MODE)
        if key not in all_data:
            continue
        df = all_data[key]
        if col not in df.columns:
            continue
        for group_val, group_df in df.groupby(col):
            refs = group_df["reference"].fillna("").tolist()
            hyps = group_df["hypothesis"].fillna("").tolist()
            stats = compute_corpus_wer(refs, hyps)
            region_data.setdefault(group_val, {})[model] = round(stats["corpus_wer"] * 100, 2)

    if not region_data:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(region_data.keys())
    x = range(len(groups))

    for i, model in enumerate(MODELS):
        values = [region_data.get(g, {}).get(model, 0) for g in groups]
        offset = (i - (len(MODELS) - 1) / 2) * bar_width
        ax.bar([xi + offset for xi in x], values, bar_width, label=MODEL_DISPLAY.get(model, model))

    ax.set_xlabel(col)
    ax.set_ylabel("WER (%)")
    ax.set_title(f"WER by {col} (mode: {PRIMARY_MODE})")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ANALYSIS_DIR, chart_name)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")

# --------------- 4. Markdown summary report ---------------
print(f"\n{'=' * 70}")
print("GENERATING SUMMARY REPORT")
print("=" * 70)

report_lines = [
    "# WER Evaluation Summary -- TIE_shorts (Indian English)",
    "",
    "## Corpus-level WER (%) by Model and Evaluation Mode",
    "",
    build_md_table(df_summary),
    "",
    "## Evaluation Modes",
    "",
    "| Mode | Reference | Before norm | After norm | Purpose |",
    "|------|-----------|-------------|------------|---------|",
    "| `transcript_raw` | Transcript | as-is | as-is | Upper bound baseline |",
    "| `transcript_clean` | Transcript | Transcript | normalized | Gold standard — paper primary |",
    "| `hf_raw` | Normalised_Transcript | as-is | as-is | HuggingFace normalization as-is |",
    "| `hf_clean` | Normalised_Transcript | Normalised_Transcript | normalized | HF + our normalizer |",
    "",
    "## Normalization Notes",
    "",
    "- `transcript_clean` is the gold standard: uses original ground truth with correct forward normalization.",
    "- `hf_raw` and `hf_clean` show the impact of the dataset's broken `Normalised_Transcript` (e.g. '1st' → 'one s t').",
    "- All modes are **symmetric**: same normalization applied to both reference and hypothesis.",
    "- Normalization: lowercase + expand contractions + fix possessives + digits/ordinals → words (num2words).",
    "",
    "## Column Schema",
    "",
    "Each result CSV contains:",
    "`split, ID, Speaker_ID, Gender, Speech_Class, Native_Region, Speech_Duration_seconds,`",
    "`Discipline_Group, Topic, model, mode, reference_source, reference_raw, reference,`",
    "`hypothesis_raw, hypothesis, wer`",
    "",
    "- `reference_raw`: original Transcript before normalization (for manual verification)",
    "- `reference`: text used for WER after normalization",
    "- `hypothesis_raw`: raw ASR output before normalization",
    "- `hypothesis`: ASR output after normalization",
    "",
]

report_lines.append("## Best Model per Mode")
report_lines.append("")
for mode in MODES:
    valid = df_summary[df_summary[mode].notna()]
    if valid.empty:
        continue
    best = valid.loc[valid[mode].idxmin()]
    report_lines.append(f"- **{mode}**: {MODEL_DISPLAY.get(best['model'], best['model'])} ({best[mode]:.2f}%)")
report_lines.append("")

report_path = os.path.join(ANALYSIS_DIR, "summary_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"  Saved: {report_path}")

# --------------- 5. Top 20 highest WER per model per mode ---------------
print(f"\n{'=' * 70}")
print("TOP 20 HIGHEST WER SENTENCES -- ALL MODELS x ALL MODES")
print("=" * 70)

for mode_name in MODES:
    top20_mode_all = []
    for model_name in MODELS:
        top20_path = os.path.join(STAGE2_DIR, f"top_20_high_wer_{model_name}_{mode_name}.csv")
        if not os.path.exists(top20_path):
            key = (model_name, mode_name)
            if key not in all_data:
                continue
            df = all_data[key].copy()
            df_sorted = df.sort_values("wer", ascending=False).head(20)
        else:
            df_sorted = pd.read_csv(top20_path)

        df_sorted = df_sorted.copy()
        if "model" not in df_sorted.columns:
            df_sorted.insert(0, "model", model_name)
        if "rank" not in df_sorted.columns:
            df_sorted.insert(1, "rank", range(1, len(df_sorted) + 1))
        top20_mode_all.append(df_sorted)

        print(f"\n  --- {MODEL_DISPLAY.get(model_name, model_name)} | mode: {mode_name} ---")
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"  #{i:2d}  ID: {row['ID']}  WER: {row['wer']:.4f} ({row['wer']*100:.1f}%)  "
                  f"Region: {row.get('Native_Region', 'N/A')}  Class: {row.get('Speech_Class', 'N/A')}")

    if top20_mode_all:
        df_combined = pd.concat(top20_mode_all, ignore_index=True)
        out_path = os.path.join(ANALYSIS_DIR, f"top_20_high_wer_all_models_{mode_name}.csv")
        df_combined.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

print("\nDone.")
