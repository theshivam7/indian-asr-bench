"""WER computation utilities."""

import statistics

import jiwer


def compute_sample_wer(ref: str, hyp: str) -> float:
    """Compute WER for a single reference-hypothesis pair.

    WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words

    Returns 1.0 if hypothesis is empty (all ref words count as deletions).
    Returns 0.0 if both ref and hyp are empty.
    """
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return jiwer.wer(ref, hyp)


def compute_corpus_wer(
    refs: list[str],
    hyps: list[str],
    per_sample_wers: list[float] | None = None,
) -> dict:
    """Compute corpus-level WER, handling empty hypotheses as all-deletion errors.

    If per_sample_wers is provided, also returns distribution stats
    (mean, median, std, p90, p95).
    """
    valid_refs = [r for r, h in zip(refs, hyps) if h]
    valid_hyps = [h for h in hyps if h]

    if valid_refs:
        output = jiwer.process_words(valid_refs, valid_hyps)
        corpus_errors = output.substitutions + output.deletions + output.insertions
    else:
        corpus_errors = 0

    empty_ref_words = sum(len(r.split()) for r, h in zip(refs, hyps) if not h)
    total_ref_words = sum(len(r.split()) for r in refs)
    corpus_errors += empty_ref_words

    corpus_wer = corpus_errors / total_ref_words if total_ref_words else 0.0

    result = {
        "corpus_wer": corpus_wer,
        "total_ref_words": total_ref_words,
        "total_errors": corpus_errors,
        "num_samples": len(refs),
        "num_empty_hyps": len(refs) - len(valid_refs),
    }

    if per_sample_wers and len(per_sample_wers) > 0:
        sorted_wers = sorted(per_sample_wers)
        n = len(sorted_wers)
        result["mean_wer"] = statistics.mean(sorted_wers)
        result["median_wer"] = statistics.median(sorted_wers)
        result["std_wer"] = statistics.stdev(sorted_wers) if n > 1 else 0.0
        result["p90_wer"] = sorted_wers[int(n * 0.9)]
        result["p95_wer"] = sorted_wers[int(n * 0.95)]

    return result
