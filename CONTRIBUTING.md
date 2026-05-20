# Contributing to Indian-ASR-Bench

Thank you for your interest in contributing.

## Ways to Contribute

- **Bug reports** — open a GitHub issue with the error message and steps to reproduce
- **New model evaluations** — add a new task directory following the existing pattern
- **Normalization improvements** — extend `utils/normalize.py`
- **Dataset extensions** — evaluate on additional splits or related datasets
- **Documentation** — improve setup instructions, add examples

## Adding a New Model

1. Create `task6_yourmodel/` with:
   - `wer_yourmodel.py` — transcription script following the existing task pattern
   - `requirements.txt` — pinned dependencies
   - `setup.sh` — environment setup
2. Add `"yourmodel"` to the `MODELS` tuple in `normalize_and_score.py` and `analysis/compare_all.py`
3. Run `python normalize_and_score.py` and `python analysis/compare_all.py` to generate results
4. Update `README.md` with the new results

The transcription script must:
- Save to `results/stage1_raw_transcripts/wer_yourmodel_raw.csv`
- Include checkpoint/resume logic (see any existing task for reference)
- Use `build_sample_row()` from `utils.io_helpers` for the output row schema

## Code Style

- Python 3.10+, PEP 8
- Type hints on all function signatures
- Docstrings on all public functions

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make changes, verify with `python normalize_and_score.py`
3. Run a quick syntax check: `python -m py_compile task*/wer_*.py utils/*.py`
4. Open a PR with a clear description of what changed and why

## Reporting Issues

Please include:
- Python version and OS
- Which task/script failed
- Full error traceback
- Contents of the relevant `requirements.txt`
