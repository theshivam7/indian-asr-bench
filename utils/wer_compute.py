"""WER / CER computation utilities and per-sample error-diagnostic helpers."""

import statistics

import jiwer


def compute_sample_cer(ref: str, hyp: str) -> float:
    """Character error rate for a single pair. Same empty-handling as WER."""
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    return jiwer.cer(ref, hyp)


def reference_word_recall(ref: str, hyp: str) -> float:
    """Fraction of *reference* word types that appear anywhere in the hypothesis.

    High recall + long hypothesis => "clip over-run" (model transcribed the
    reference correctly PLUS extra real speech the clip cut off). Low recall =>
    "content mismatch" (audio does not match the reference). Central to the
    artifact taxonomy in analysis/error_analysis.py.
    """
    ref_words = set(ref.split())
    if not ref_words:
        return 0.0
    hyp_words = set(hyp.split())
    return len(ref_words & hyp_words) / len(ref_words)


def length_ratio(ref: str, hyp: str) -> float:
    """Hypothesis/reference word-count ratio. >1 => hypothesis longer (insertion-heavy)."""
    n_ref = len(ref.split())
    if n_ref == 0:
        return 0.0
    return len(hyp.split()) / n_ref


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
        subs, dels, ins = output.substitutions, output.deletions, output.insertions
        corpus_errors = subs + dels + ins
    else:
        subs = dels = ins = 0
        corpus_errors = 0

    empty_ref_words = sum(len(r.split()) for r, h in zip(refs, hyps) if not h)
    total_ref_words = sum(len(r.split()) for r in refs)
    # Empty hypotheses count as all-deletion errors.
    dels += empty_ref_words
    corpus_errors += empty_ref_words

    corpus_wer = corpus_errors / total_ref_words if total_ref_words else 0.0

    result = {
        "corpus_wer": corpus_wer,
        "total_ref_words": total_ref_words,
        "total_errors": corpus_errors,
        "num_samples": len(refs),
        "num_empty_hyps": len(refs) - len(valid_refs),
        # S/D/I breakdown (additive; corpus_wer above is unchanged). Insertion rate
        # = insertions / total_ref_words is the corpus-level hallucination signal.
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "insertion_rate": ins / total_ref_words if total_ref_words else 0.0,
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


def compute_corpus_cer(refs: list[str], hyps: list[str]) -> float:
    """Corpus character error rate: total char edits / total reference chars.

    Empty hypotheses count all reference characters as deletions (mirrors the WER
    empty-handling), so CER and WER treat missing output consistently.
    """
    valid_refs = [r for r, h in zip(refs, hyps) if h]
    valid_hyps = [h for h in hyps if h]

    if valid_refs:
        output = jiwer.process_characters(valid_refs, valid_hyps)
        char_errors = output.substitutions + output.deletions + output.insertions
    else:
        char_errors = 0

    char_errors += sum(len(r) for r, h in zip(refs, hyps) if not h)
    total_ref_chars = sum(len(r) for r in refs)
    return char_errors / total_ref_chars if total_ref_chars else 0.0
