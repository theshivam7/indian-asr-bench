"""Shared utilities.

Scripts import the submodules they need directly (e.g. `from utils.normalize import ...`,
`from utils.transcribe import transcribe_sample`), so this package intentionally does NOT
eagerly re-export everything. Doing so would force the CPU-only Stage 2/3 pipeline
(normalize_and_score.py, analysis/) to import the audio/transcription stack (librosa,
torch, datasets) just to recompute WER or draw charts. Import submodules directly instead.
"""
