"""Central registry — the single source of truth for the benchmark.

Everything the pipeline needs to know about **models**, **datasets**, and
**evaluation modes** lives here and *only* here. Every other module imports from
this file; no model list, dataset id, mode name, display string, colour, or
metadata-column list is defined anywhere else.

This module is deliberately **pure Python** — it imports nothing heavy
(no torch / datasets / whisper / jiwer). Both the CPU-only analysis pipeline and
the GPU inference scripts can import it safely.

Adding a new model  -> append one ModelSpec to MODEL_SPECS.
Adding a new dataset -> append one DatasetSpec to DATASET_SPECS.
Adding a new mode    -> append one ModeSpec to MODE_SPECS.
"""

from dataclasses import dataclass, field


# ============================================================================
# Evaluation modes
# ============================================================================
# A *mode* = (reference source, normalizer), applied symmetrically to reference
# and hypothesis. `reference` is a canonical role ("gold" or "alt"); the dataset
# adapter maps that role onto the actual column stored in the raw CSV
# (transcript_raw / normalised_transcript_raw). `normalizer` selects the text
# cleanup applied by utils.normalize.

@dataclass(frozen=True)
class ModeSpec:
    key: str          # canonical mode key used in filenames + columns
    reference: str    # "gold" (Transcript) or "alt" (Normalised_Transcript)
    normalizer: str   # "minimal" | "custom" | "whisper"
    display: str      # short label for tables/figures


MODE_SPECS = (
    ModeSpec("transcript_raw",   "gold", "minimal", "T-raw"),
    ModeSpec("transcript_clean", "gold", "custom",  "T-clean"),   # <- PRIMARY / gold
    ModeSpec("hf_raw",           "alt",  "minimal", "HF-raw"),
    ModeSpec("hf_clean",         "alt",  "custom",  "HF-clean"),
    ModeSpec("whisper_norm",     "gold", "whisper", "Whisper-norm"),  # community-standard comparison
)
MODE_BY_KEY = {m.key: m for m in MODE_SPECS}
ALL_MODES = tuple(m.key for m in MODE_SPECS)
PRIMARY_MODE = "transcript_clean"


def get_reference_role(mode: str) -> str:
    if mode not in MODE_BY_KEY:
        raise ValueError(f"Unknown mode: {mode}. Valid: {ALL_MODES}")
    return MODE_BY_KEY[mode].reference


def get_normalizer(mode: str) -> str:
    if mode not in MODE_BY_KEY:
        raise ValueError(f"Unknown mode: {mode}. Valid: {ALL_MODES}")
    return MODE_BY_KEY[mode].normalizer


# ============================================================================
# Models
# ============================================================================
# `engine`     selects the inference backend / transcribe function.
# `arch_class` is scientifically load-bearing for the artifact diagnostic:
#              CTC / transducer models emit only acoustically-grounded tokens
#              (cannot free-run/hallucinate), so they are the "witnesses" that
#              prove over-run content is real speech. enc_dec (Whisper) and llm
#              (Qwen3) *can* hallucinate.
# `chart`      whether the model appears in the headline ranking charts (the
#              FT-study variants are excluded to keep the engine comparison fair).
# `tie_only`   FT variants are TIE-specific (Svarah is eval-only, no fine-tune).

ENGINES = ("openai_whisper", "hf_whisper", "nemo_tdt", "nemo_ctc", "qwen")
ARCH_CLASSES = ("enc_dec", "transducer", "ctc", "llm")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    engine: str
    model_id: str        # checkpoint / load id for the engine
    env: str             # conda env name (see environments/)
    arch_class: str
    params: str          # human-readable parameter count
    color: str           # Okabe-Ito hex, fixed across all figures
    order: int           # sort order in tables/figures
    chart: bool = True
    tie_only: bool = False


# Okabe-Ito colourblind-safe palette (kept identical to the previous figures):
#   blue #0072B2 · orange #E69F00 · green #009E73 · vermillion #D55E00 ·
#   reddish-purple #CC79A7 · sky-blue #56B4E9 · black #000000
MODEL_SPECS = (
    ModelSpec("base",   "Whisper Base",   "openai_whisper", "base",   "whisper", "enc_dec",    "74M",   "#0072B2", 10),
    ModelSpec("medium", "Whisper Medium", "openai_whisper", "medium", "whisper", "enc_dec",    "769M",  "#E69F00", 20),
    ModelSpec("large",  "Whisper Large",  "openai_whisper", "large",  "whisper", "enc_dec",    "1.5B",  "#009E73", 30),
    ModelSpec("large_v3_turbo", "Whisper large-v3-turbo", "openai_whisper", "turbo", "whisper", "enc_dec", "809M", "#56B4E9", 35),
    ModelSpec("parakeet",     "Parakeet-TDT-0.6B", "nemo_tdt", "nvidia/parakeet-tdt-0.6b-v2", "parakeet", "transducer", "600M",  "#D55E00", 40),
    ModelSpec("parakeet_ctc", "Parakeet-CTC-1.1B", "nemo_ctc", "nvidia/parakeet-ctc-1.1b",    "parakeet", "ctc",        "1.1B",  "#882255", 45),
    ModelSpec("qwen3",  "Qwen3-ASR-1.7B", "qwen", "Qwen/Qwen3-ASR-1.7B", "qwen3", "llm", "1.7B", "#CC79A7", 50),
    # --- Fine-tuning study variants (TIE-only, excluded from headline charts) ---
    ModelSpec("medium_hf", "Whisper Medium (HF)", "hf_whisper", "openai/whisper-medium", "whisper_medium_ft", "enc_dec", "769M", "#E69F00", 60, chart=False, tie_only=True),
    ModelSpec("medium_ft", "Whisper Medium (FT)", "hf_whisper", "models/whisper_medium_ft", "whisper_medium_ft", "enc_dec", "769M", "#B07200", 70, chart=False, tie_only=True),
    ModelSpec("medium_ft_disjoint", "Whisper Medium (FT, speaker-disjoint)", "hf_whisper", "models/whisper_medium_ft_disjoint", "whisper_medium_ft", "enc_dec", "769M", "#7A4E00", 80, chart=False, tie_only=True),
    # Seed replicates of the disjoint FT (seeds 43/44; the entry above is seed 42). The
    # null result ("FT gains vanish once speaker leakage is removed") is only credible if
    # it is stable across training seeds — Whisper FT seed variance is the same order as
    # the effect being denied. Excluded from charts; aggregated in compare_finetune.py.
    ModelSpec("medium_ft_disjoint_s43", "Whisper Medium (FT, disjoint, seed 43)", "hf_whisper", "models/whisper_medium_ft_disjoint_s43", "whisper_medium_ft", "enc_dec", "769M", "#6B4400", 81, chart=False, tie_only=True),
    ModelSpec("medium_ft_disjoint_s44", "Whisper Medium (FT, disjoint, seed 44)", "hf_whisper", "models/whisper_medium_ft_disjoint_s44", "whisper_medium_ft", "enc_dec", "769M", "#5C3A00", 82, chart=False, tie_only=True),
)
MODEL_BY_KEY = {m.key: m for m in MODEL_SPECS}
ALL_MODELS = tuple(m.key for m in MODEL_SPECS)

# Convenience views (derived, never hand-maintained)
CHART_MODELS = tuple(m.key for m in sorted(MODEL_SPECS, key=lambda m: m.order) if m.chart)
MODEL_DISPLAY = {m.key: m.display for m in MODEL_SPECS}
MODEL_COLOR = {m.key: m.color for m in MODEL_SPECS}
MODEL_ORDER = [m.key for m in sorted(MODEL_SPECS, key=lambda m: m.order)]


# ============================================================================
# Datasets
# ============================================================================
# The dataset adapter (utils/datasets.py) uses `column_map` to translate each
# dataset's raw HF columns into the canonical raw-CSV schema, so everything after
# Stage 1 is dataset-agnostic. `subgroup_dims` drives the Stage-3 breakdown
# tables (each entry = (raw_csv_column, display_label)). `applicable_modes` gates
# which evaluation modes make sense (Svarah has no pre-normalized field, so the
# hf_* modes are excluded).

@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    display: str
    splits: dict                 # role -> HF split name, e.g. {"eval": "test", "train": "train"}
    gold_ref_col: str            # HF column for the gold reference  -> canonical transcript_raw
    alt_ref_col: str | None      # HF column for the alt reference   -> canonical normalised_transcript_raw (None if absent)
    id_col: str                  # HF column used as the per-clip ID
    speaker_col: str | None      # HF column with a speaker id (for leakage / speaker analysis)
    audio_col: str               # HF column holding audio
    duration_col: str | None     # HF column with clip duration in seconds (None -> derived from audio)
    metadata_cols: dict          # canonical_output_name -> HF source column (extra columns carried into the raw CSV)
    subgroup_dims: tuple         # ((csv_col, display), ...) for Stage-3 breakdowns (duration handled separately)
    applicable_modes: tuple      # subset of ALL_MODES valid for this dataset
    license: str
    citation: str
    hf_revision: str | None = None   # pinned HF dataset commit sha (reproducibility; None = latest)
    neer_register_col: str | None = None   # csv column whose value selects the entity-dense register (Svarah use-cases)
    neer_register_value: str | None = None
    verified: bool = True        # False -> column_map is provisional, adapter must confirm against ds.features
    audio_undecoded: bool = False  # True -> adapter casts audio_col to Audio(decode=False) on load,
    #                               so accessing it yields the raw {"bytes","path"} storage dict and
    #                               datasets' decode machinery (torchcodec-mandatory in datasets>=4,
    #                               but no torchcodec release satisfies both torch==2.5.1 and
    #                               datasets>=4) is never invoked. Engines decode via
    #                               utils.io_helpers.decode_audio_value (soundfile) instead.
    #                               Keep False for TIE: its "audio" column stores raw float arrays
    #                               (not Audio-typed bytes), which already bypasses the decoder.


TIE = DatasetSpec(
    key="tie",
    hf_id="raianand/TIE_shorts",
    display="TIE_shorts",
    splits={"train": "train", "validation": "validation", "eval": "test"},
    gold_ref_col="Transcript",
    alt_ref_col="Normalised_Transcript",
    id_col="ID",
    speaker_col="Speaker_ID",
    audio_col="audio",
    duration_col="Speech_Duration_seconds",
    metadata_cols={
        "Gender": "Gender",
        "Speech_Class": "Speech_Class",
        "Native_Region": "Native_Region",
        "Discipline_Group": "Discipline_Group",
        "Topic": "Topic",
    },
    subgroup_dims=(
        ("Native_Region", "Region"),
        ("Discipline_Group", "Discipline"),
        ("Speech_Class", "Speech rate"),
        ("Gender", "Gender"),
    ),
    applicable_modes=("transcript_raw", "transcript_clean", "hf_raw", "hf_clean", "whisper_norm"),
    license="CC BY-SA 2.0",
    citation="Rai et al., ICWSM 2024 (NPTEL-derived)",
    hf_revision="28c53e285feae86f4ba25d8aaeca4fd0c709784c",  # 2024-11-16; predates all runs
    neer_register_col=None,     # academic prose: no entity-dense register -> NEER not applicable
)

# Confirmed against the real ai4bharat/Svarah [test] HF features on first load
# (2026-07-03): ['age-group', 'audio_filepath', 'duration', 'gender',
# 'highest_qualification', 'job_category', 'native_place_district',
# 'native_place_state', 'occupation_domain', 'primary_language', 'text'].
# No speaker-id column and no read/extempore/use-case register column are exposed
# in this Hub config, despite the dataset card describing that split conceptually
# -> speaker_col is None and NEER (register-gated) is disabled until a real
# register field is found (e.g. derivable from audio_filepath naming, or from the
# original AI4Bharat release rather than this HF mirror).
SVARAH = DatasetSpec(
    key="svarah",
    hf_id="ai4bharat/Svarah",
    display="Svarah",
    splits={"eval": "test"},                 # eval-only benchmark (no train/val)
    gold_ref_col="text",
    alt_ref_col=None,                         # no pre-normalized reference field
    id_col="audio_filepath",
    speaker_col=None,                         # no speaker-id column in this HF config
    audio_col="audio_filepath",
    duration_col="duration",
    metadata_cols={
        "Gender": "gender",
        "Age": "age-group",
        "Native_Language": "primary_language",
    },
    subgroup_dims=(
        ("Native_Language", "Native language"),
        ("Gender", "Gender"),
        ("Age", "Age group"),
    ),
    applicable_modes=("transcript_raw", "transcript_clean", "whisper_norm"),
    license="CC BY 4.0",
    citation="Javed et al., INTERSPEECH 2023",
    hf_revision="ebbf7777fe771490696a3f7b007097606fa8c924",  # 2025-03-10
    neer_register_col=None,                   # no register field in this HF config; see note above
    neer_register_value=None,
    verified=True,
    audio_undecoded=True,   # bytes-stored audio: bypass datasets' (torchcodec) decoder entirely
)

DATASET_SPECS = (TIE, SVARAH)
DATASET_BY_KEY = {d.key: d for d in DATASET_SPECS}


def get_dataset(key: str) -> DatasetSpec:
    if key not in DATASET_BY_KEY:
        raise ValueError(f"Unknown dataset: {key}. Valid: {tuple(DATASET_BY_KEY)}")
    return DATASET_BY_KEY[key]


def models_for_dataset(dataset_key: str) -> tuple:
    """Model keys applicable to a dataset (excludes TIE-only FT variants elsewhere)."""
    d = get_dataset(dataset_key)
    if d.key == "tie":
        return ALL_MODELS
    return tuple(m.key for m in MODEL_SPECS if not m.tie_only)


def modes_for_dataset(dataset_key: str) -> tuple:
    return get_dataset(dataset_key).applicable_modes
