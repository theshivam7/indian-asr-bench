# Contributing to Indian-ASR-Bench

Thank you for your interest in contributing.

## Ways to Contribute

- **Bug reports**: open a GitHub issue with the error message and steps to reproduce
- **New model evaluations**: add a new task directory following the existing pattern
- **Normalization improvements**: extend `utils/normalize.py`
- **Dataset extensions**: evaluate on additional splits or related datasets
- **Documentation**: improve setup instructions, add examples

## Framework overview

`utils/registry.py` is the single source of truth for models, datasets, and
evaluation modes; nothing about them is defined anywhere else. The pipeline is
three dataset-namespaced stages: Stage 1 inference → `results/<dataset>/stage1_raw_transcripts/`
(immutable, committed), Stage 2 `normalize_and_score.py` → `results/<dataset>/stage2_processed/`,
Stage 3 `analysis/*` + `paper/figures/` → `results/<dataset>/analysis/`.

## Adding a New Model

1. Append one `ModelSpec` to `MODEL_SPECS` in `utils/registry.py` (key, display,
   engine, `model_id`, conda env, `arch_class`, params, a colourblind-safe colour
   validated with the dataviz palette checker, and sort order).
2. Add its inference path:
   - reuse an existing engine driver if the engine matches (`whisper_asr/run_whisper.py`,
     `parakeet/wer_parakeet.py`, `qwen3/wer_qwen3.py`; all take `--model`/`--dataset`), **or**
   - add a new `taskN_yourmodel/` (driver + `requirements.txt` + `setup.sh`) that calls
     `utils.inference_loop.run_transcription(model_key, dataset_key, transcribe_one)`.
3. Run inference, then `python normalize_and_score.py --dataset <ds>` and the
   `analysis/*` scripts (all `--dataset`-aware) to regenerate every table/figure.
4. Update `README.md` with the new results.

The transcription output must land at
`results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv` (use the
`utils.io_helpers` path + `build_sample_row()` helpers; the shared inference loop
does this for you) and be committed: raw transcripts are the immutable source of
truth, so any later normalization/metric change recomputes without re-inference.

## Adding a New Dataset

Append one `DatasetSpec` to `utils/registry.py` (HF id, canonical column map,
subgroup dims, applicable modes). No other file changes; the adapter
(`utils/datasets.py`) validates the schema and everything after Stage 1 is
dataset-agnostic.

## Code Style

- Python 3.10+, PEP 8
- Type hints on all function signatures
- Docstrings on all public functions

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make changes, verify with `python normalize_and_score.py --dataset tie` (should reproduce committed numbers)
3. Run the tests: `python tests/test_pipeline.py` (or `python -m pytest tests/ -q`); this pins the
   normalization/WER contracts, registry integrity, and the committed headline numbers
4. Run a quick syntax check: `python -m py_compile utils/*.py analysis/*.py whisper_asr/*.py task*/wer_*.py`
4. Open a PR with a clear description of what changed and why

## Reporting Issues

Please include:
- Python version and OS
- Which task/script failed
- Full error traceback
- Contents of the relevant `requirements.txt`
