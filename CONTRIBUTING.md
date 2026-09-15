# Contributing to Indian-ASR-Bench

Thank you for your interest in contributing.

## Ways to contribute

- **Bug reports**: open a GitHub issue with the error message and steps to reproduce
- **New model evaluations**: add a new engine directory following the existing pattern
- **Normalization improvements**: extend `utils/normalize.py`
- **Dataset extensions**: evaluate on additional splits or related datasets
- **Documentation**: improve setup instructions, add examples

## Framework overview

`utils/registry.py` is the single source of truth for models, datasets, and
evaluation modes; nothing about them is defined anywhere else. The pipeline has
three dataset-namespaced stages:

1. Stage 1 inference writes to `results/<dataset>/stage1_raw_transcripts/` (immutable, committed).
2. Stage 2 `normalize_and_score.py` writes to `results/<dataset>/stage2_processed/`.
3. Stage 3 `analysis/*` writes to `results/<dataset>/analysis/`.

## Adding a new model

1. Append one `ModelSpec` to `MODEL_SPECS` in `utils/registry.py` (key, display,
   engine, `model_id`, `env`, `arch_class`, params, a colorblind-safe color, and sort order).
2. Add its inference path:
   - reuse an existing engine driver if the engine matches (`whisper_asr/run_whisper.py`,
     `parakeet/wer_parakeet.py`, `qwen3/wer_qwen3.py`; all take `--model` and `--dataset`), or
   - add a new engine directory (like `whisper_asr/`, `parakeet/`, `qwen3/`) with a driver,
     a `requirements.txt` and a `setup.sh`. The driver calls
     `utils.inference_loop.run_transcription(model_key, dataset_key, transcribe_one)`.
3. Run inference, then `python normalize_and_score.py --dataset <ds>` and the
   `analysis/*` scripts (all take `--dataset`) to regenerate every table and figure.
4. Update `README.md` and `SUMMARY.md` with the new results.

The transcription output must land at
`results/<dataset>/stage1_raw_transcripts/wer_<model>_raw.csv` (the shared inference
loop does this for you) and be committed. Raw transcripts are the immutable source of
truth, so any later normalization or metric change recomputes without re-inference.

## Adding a new dataset

Append one `DatasetSpec` to `utils/registry.py` (HF id, column names, subgroup
dims, applicable modes). No other file changes; the adapter (`utils/datasets.py`)
validates the schema, and everything after Stage 1 is dataset-agnostic.

## Code style

- Python 3.10+, PEP 8
- Type hints on all function signatures
- Docstrings on all public functions
- Plain ASCII in prose and generated reports: no em dashes, arrows or emoji

## Pull request process

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Per-clip Stage 2 CSVs are not tracked, so run `python normalize_and_score.py --dataset <ds>`
   before any `analysis/` script. On TIE this should reproduce the committed numbers.
3. Run the tests: `python -m pytest tests/ -q`. They pin the normalization and WER
   contracts, registry integrity, and the committed headline numbers.
4. Open a PR with a clear description of what changed and why.

## Reporting issues

Please include:

- Python version and OS
- Which script failed
- Full error traceback
- Contents of the relevant `requirements.txt`
