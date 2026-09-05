"""Central registry, the single source of truth for the benchmark.

Everything the pipeline needs to know about **models**, **datasets**, and
**evaluation modes** lives here and *only* here. Every other module imports from
this file; no model list, dataset id, mode name, display string, colour, or
metadata-column list is defined anywhere else.

This module is deliberately **pure Python**, it imports nothing heavy
(no torch / datasets / whisper / jiwer). Both the CPU-only analysis pipeline and
the GPU inference scripts can import it safely.

Adding a new model  -> append one ModelSpec to MODEL_SPECS.
Adding a new dataset -> append one DatasetSpec to DATASET_SPECS.
Adding a new mode    -> append one ModeSpec to MODE_SPECS.
"""

from dataclasses import dataclass


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
# `only_datasets`  restricts a model to specific dataset keys (None = all datasets).
#              Fine-tuned variants only apply to the dataset they were trained on;
#              their HF-pipeline pretrained baselines apply to every fine-tunable
#              dataset (Svarah is eval-only, no fine-tune).

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
    only_datasets: tuple | None = None


# Okabe-Ito colourblind-safe palette (kept identical to the previous figures):
#   blue #0072B2 · orange #E69F00 · green #009E73 · vermillion #D55E00 ·
#   reddish-purple #CC79A7 · sky-blue #56B4E9 · black #000000
MODEL_SPECS = (
    ModelSpec("tiny",   "Whisper Tiny",   "openai_whisper", "tiny",   "whisper", "enc_dec",    "39M",   "#000000", 5),
    ModelSpec("base",   "Whisper Base",   "openai_whisper", "base",   "whisper", "enc_dec",    "74M",   "#0072B2", 10),
    ModelSpec("small",  "Whisper Small",  "openai_whisper", "small",  "whisper", "enc_dec",    "244M",  "#F0E442", 15),
    ModelSpec("medium", "Whisper Medium", "openai_whisper", "medium", "whisper", "enc_dec",    "769M",  "#E69F00", 20),
    # model_id is explicit "large-v3" rather than the "large" alias: functionally identical
    # (openai-whisper's "large" alias resolves to large-v3 as of the pinned package version),
    # but explicit is more robust against the alias target changing in a future release.
    ModelSpec("large",  "Whisper Large-v3",  "openai_whisper", "large-v3",  "whisper", "enc_dec",    "1.5B",  "#009E73", 30),
    ModelSpec("large_v3_turbo", "Whisper large-v3-turbo", "openai_whisper", "turbo", "whisper", "enc_dec", "809M", "#56B4E9", 35),
    ModelSpec("parakeet",     "Parakeet-TDT-0.6B-v2", "nemo_tdt", "nvidia/parakeet-tdt-0.6b-v2", "parakeet", "transducer", "600M",  "#D55E00", 40),
    ModelSpec("parakeet_ctc", "Parakeet-CTC-1.1B", "nemo_ctc", "nvidia/parakeet-ctc-1.1b",    "parakeet", "ctc",        "1.1B",  "#882255", 45),
    ModelSpec("qwen3",  "Qwen3-ASR-1.7B", "qwen", "Qwen/Qwen3-ASR-1.7B", "qwen3", "llm", "1.7B", "#CC79A7", 50),
    # --- Fine-tuning study variants (excluded from headline charts). The *_hf baselines
    # apply to every fine-tunable dataset; the *_ft variants only to the dataset they were
    # trained on. ---
    ModelSpec("medium_hf", "Whisper Medium (HF)", "hf_whisper", "openai/whisper-medium", "whisper_medium_ft", "enc_dec", "769M", "#E69F00", 60, chart=False, only_datasets=("tie", "aesrc")),
    ModelSpec("medium_ft", "Whisper Medium (FT)", "hf_whisper", "models/whisper_medium_ft", "whisper_medium_ft", "enc_dec", "769M", "#B07200", 70, chart=False, only_datasets=("tie",)),
    # --- TIE capacity study: Tiny/Small fine-tuning (follow-up to the Medium null result;
    # see results/tie/analysis/findings_tiny_small_ft.md). Official-split only. ---
    ModelSpec("tiny_hf",  "Whisper Tiny (HF)",  "hf_whisper", "openai/whisper-tiny",     "whisper_medium_ft", "enc_dec", "39M",  "#4D4D4D", 90, chart=False, only_datasets=("tie", "aesrc")),
    ModelSpec("tiny_ft",  "Whisper Tiny (FT)",  "hf_whisper", "models/whisper_tiny_ft",  "whisper_medium_ft", "enc_dec", "39M",  "#7F7F7F", 91, chart=False, only_datasets=("tie",)),
    ModelSpec("small_hf", "Whisper Small (HF)", "hf_whisper", "openai/whisper-small",    "whisper_medium_ft", "enc_dec", "244M", "#B8A73A", 92, chart=False, only_datasets=("tie", "aesrc")),
    ModelSpec("small_ft", "Whisper Small (FT)", "hf_whisper", "models/whisper_small_ft", "whisper_medium_ft", "enc_dec", "244M", "#8A7B1F", 93, chart=False, only_datasets=("tie",)),
    # --- AESRC capacity study: Tiny/Small/Medium fine-tuned on the AESRC2020 Indian train
    # split (natively speaker-disjoint test set). All three use the step-based recipe in
    # finetune/finetune_tiny_small.py; baselines are the shared *_hf specs above. ---
    ModelSpec("tiny_aesrc_ft",   "Whisper Tiny (AESRC FT)",   "hf_whisper", "models/whisper_tiny_aesrc_ft",   "whisper_medium_ft", "enc_dec", "39M",  "#9A9A9A", 94, chart=False, only_datasets=("aesrc",)),
    ModelSpec("small_aesrc_ft",  "Whisper Small (AESRC FT)",  "hf_whisper", "models/whisper_small_aesrc_ft",  "whisper_medium_ft", "enc_dec", "244M", "#6B5F18", 95, chart=False, only_datasets=("aesrc",)),
    ModelSpec("medium_aesrc_ft", "Whisper Medium (AESRC FT)", "hf_whisper", "models/whisper_medium_aesrc_ft", "whisper_medium_ft", "enc_dec", "769M", "#8F5C00", 96, chart=False, only_datasets=("aesrc",)),
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
    cluster_id_regex: str | None = None  # regex with ONE capture group applied to the clip ID to
    #                               recover a resampling-cluster tag when no speaker column exists.
    #                               Used by analysis/statistics.py as the bootstrap cluster unit.
    #                               Document per dataset what the tag actually is (recording vs
    #                               speaker), the statistics report states it verbatim.
    audio_undecoded: bool = False  # True -> adapter casts audio_col to Audio(decode=False) on load,
    #                               so accessing it yields the raw {"bytes","path"} storage dict and
    #                               datasets' decode machinery (torchcodec-mandatory in datasets>=4,
    #                               but no torchcodec release satisfies both torch==2.5.1 and
    #                               datasets>=4) is never invoked. Engines decode via
    #                               utils.io_helpers.decode_audio_value (soundfile) instead.
    #                               Keep False for TIE: its "audio" column stores raw float arrays
    #                               (not Audio-typed bytes), which already bypasses the decoder.
    filter_col: str | None = None  # HF column + value defining the subset of rows this spec uses
    filter_value: str | None = None  # (e.g. accent == "INDIAN"). Applied by the adapter on every
    #                               split load, so all pipeline stages see only the subset.


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
    # audio_filepath embeds a recording/file tag: "<num>_f2235_chunk_24.wav" -> f2235.
    # Verified 2026-07-03: 3232 distinct tags over 6656 clips, each tag demographically
    # consistent (never >1 gender/age/native-language), chunks of one recording share it.
    # This is a RECORDING id, not a speaker id (the Svarah paper reports 117 speakers),
    # so recording-level resampling still understates within-speaker correlation, but it
    # is strictly less anti-conservative than clip-level.
    cluster_id_regex=r"_(f\d+)_chunk",
    verified=True,
    audio_undecoded=True,   # bytes-stored audio: bypass datasets' (torchcodec) decoder entirely
)

# AESRC2020 (Accented English Speech Recognition Challenge 2020, Datatang; Shi et al.,
# ICASSP 2021) via the pengyizhou/accented_english parquet mirror. The mirror carries all
# 8 accents; this spec selects the INDIAN subset only (filter_col/filter_value), verified
# against the live schema on 2026-07-16: columns id / audio (16kHz bytes-stored WAV) /
# transcription / speaker / accent; splits train (118,927) / valid (5,614) / test (14,493);
# Indian rows: 12,820 / 532 / 1,731. Test speakers (481) are fully disjoint from the 38
# train and valid speakers; valid shares train's speaker set exactly, so validation WER
# measures fit, not speaker generalization. Full population analysis (exact durations,
# speaker structure, label sanity, licensing) is in a local-only deep-dive doc, not
# committed (see docs/ in .git/info/exclude) -- the load-bearing findings from it are
# inlined above and in this spec's `license` field below.
AESRC = DatasetSpec(
    key="aesrc",
    hf_id="pengyizhou/accented_english",
    display="AESRC2020 (Indian)",
    splits={"train": "train", "validation": "valid", "eval": "test"},
    gold_ref_col="transcription",
    alt_ref_col=None,                         # no pre-normalized reference field
    id_col="id",
    speaker_col="speaker",
    audio_col="audio",
    duration_col=None,                        # no duration column; derived from audio bytes
    metadata_cols={
        "Accent": "accent",
    },
    subgroup_dims=(),                         # accent is constant after filtering; no other dims
    applicable_modes=("transcript_raw", "transcript_clean", "whisper_norm"),
    license="Unspecified (mirror carries no license; AESRC2020 is Datatang's corpus. "
            "Data-use for this study confirmed through our advisor)",
    citation="Shi et al., ICASSP 2021 (arXiv:2102.10233)",
    hf_revision="4a80d8388f06368a0fa2a325770bec3492cabd3d",  # 2026-06-29; predates all runs
    audio_undecoded=True,   # bytes-stored audio: bypass datasets' (torchcodec) decoder entirely
    filter_col="accent",
    filter_value="INDIAN",
)

DATASET_SPECS = (TIE, SVARAH, AESRC)
DATASET_BY_KEY = {d.key: d for d in DATASET_SPECS}


def get_dataset(key: str) -> DatasetSpec:
    if key not in DATASET_BY_KEY:
        raise ValueError(f"Unknown dataset: {key}. Valid: {tuple(DATASET_BY_KEY)}")
    return DATASET_BY_KEY[key]


def models_for_dataset(dataset_key: str) -> tuple:
    """Model keys applicable to a dataset, honoring each spec's only_datasets gate."""
    d = get_dataset(dataset_key)
    return tuple(m.key for m in MODEL_SPECS
                 if m.only_datasets is None or d.key in m.only_datasets)


def modes_for_dataset(dataset_key: str) -> tuple:
    return get_dataset(dataset_key).applicable_modes
