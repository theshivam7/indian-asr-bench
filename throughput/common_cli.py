"""CLI arguments shared by the three throughput drivers."""

import argparse

from utils.throughput import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_CLIPS,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_TELEMETRY_INTERVAL_MS,
    DEFAULT_WARMUP_BATCHES,
    parse_batch_sizes,
)


def parser(description: str, models: tuple[str, ...]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", required=True, choices=models)
    ap.add_argument("--dataset", required=True, choices=("tie", "svarah", "aesrc"))
    ap.add_argument("--clips", type=int, default=DEFAULT_CLIPS)
    ap.add_argument("--batch-sizes", default=",".join(map(str, DEFAULT_BATCH_SIZES)))
    ap.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--telemetry-ms", type=int, default=DEFAULT_TELEMETRY_INTERVAL_MS)
    return ap


def run_kwargs(args) -> dict:
    return {
        "batch_sizes": parse_batch_sizes(args.batch_sizes),
        "n_clips": args.clips,
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
        "seed": args.seed,
        "telemetry_interval_ms": args.telemetry_ms,
    }
