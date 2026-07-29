"""
Stage 3 (entity analysis): Named/numeric Entity Error Rate — Svarah use-cases only.

Svarah's use-case register (grocery orders, digital payments, government services)
is entity-dense: amounts, UPI IDs, bank/account numbers, brand and scheme names.
Both AI4Bharat papers show these break ASR far more than ordinary speech, but they
report WER only. We add an entity-focused metric on exactly that register.

Entity tokens are extracted from the RAW reference (before number normalization,
which would otherwise dissolve "1024"/"9876543210@paytm" into words) with a
dependency-free regex covering digit-bearing tokens, currency and alphanumeric
codes; optional spaCy NER adds person/org/place names if spaCy is installed.
NEER = 1 - (entity tokens recovered in the hypothesis / total entity tokens),
a recall-based measure that needs no alignment.

Applicable ONLY where DatasetSpec.neer_register_col is set. Currently DORMANT for
both datasets: TIE has no entity-dense register (academic prose, and its reference
artifacts would confound the measure), and the HF Svarah mirror exposes no use-case
register column (see the SVARAH spec note in utils/registry.py) — the script exits
with that explanation until a register field is derived.

Reads  results/<dataset>/stage2_processed/<mode>/wer_<model>_<mode>.csv
Writes results/<dataset>/analysis/entity_neer_<mode>.{csv,md}

Usage:
    python analysis/entity_analysis.py --dataset svarah
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.registry import PRIMARY_MODE, MODEL_DISPLAY, models_for_dataset, get_dataset
from utils.io_helpers import stage2_dir, analysis_dir, build_md_table

# Digit-bearing tokens (123, 1,024, 2kg, ₹500), currency, UPI/emails, and
# alphanumeric codes (SBI123, AB12CD). Case-insensitive; punctuation-tolerant match.
_ENTITY_RE = re.compile(
    r"""(?xi)
    (?: \d[\d,.\-/:]*\w* )          # any token starting with a digit
    | (?: \w*\d[\w@.\-]* )          # any token containing a digit (codes, UPI, emails)
    | (?: [₹$]\s?\d[\d,.]* )        # currency amounts
    """
)


def _norm_token(tok: str) -> str:
    return re.sub(r"[^\w@]", "", tok.lower())


def entity_tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    toks = [_norm_token(t) for t in _ENTITY_RE.findall(text)]
    return [t for t in toks if t]


def _spacy_entities(texts: list[str]):
    """Optional: add spaCy PERSON/ORG/GPE/PRODUCT entity tokens if spaCy is present."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        return None
    keep = {"PERSON", "ORG", "GPE", "PRODUCT", "MONEY", "CARDINAL"}
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        toks = []
        for ent in doc.ents:
            if ent.label_ in keep:
                toks += [_norm_token(t) for t in ent.text.split()]
        out.append([t for t in toks if t])
    return out


def neer_for_model(dataset: str, model: str, mode: str, register_col: str, register_val, use_spacy: bool):
    path = os.path.join(stage2_dir(dataset), mode, f"wer_{model}_{mode}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if register_col in df.columns and register_val is not None:
        df = df[df[register_col].astype(str) == str(register_val)]
    if df.empty:
        return None

    refs = df["reference_raw"].fillna("").tolist()
    hyps = df["hypothesis_raw"].fillna("").tolist()
    ref_ents = [entity_tokens(r) for r in refs]
    if use_spacy:
        extra = _spacy_entities(refs)
        if extra:
            ref_ents = [sorted(set(a) | set(b)) for a, b in zip(ref_ents, extra)]

    total, recovered = 0, 0
    for ents, hyp in zip(ref_ents, hyps):
        hset = set(entity_tokens(hyp)) | set(_norm_token(t) for t in hyp.split())
        for e in ents:
            total += 1
            if e in hset:
                recovered += 1
    if total == 0:
        return None
    return {"model": model, "display": MODEL_DISPLAY.get(model, model),
            "n_clips": len(df), "n_entities": total,
            "entity_recall_pct": round(recovered / total * 100, 2),
            "neer_pct": round((1 - recovered / total) * 100, 2)}


def main(dataset: str, mode: str, use_spacy: bool) -> None:
    spec = get_dataset(dataset)
    if not spec.neer_register_col:
        print(f"[entity_analysis] NEER not applicable to {spec.display}: no entity-dense register "
              f"(neer_register_col unset). Skipping.")
        return
    rows = [r for m in models_for_dataset(dataset)
            if (r := neer_for_model(dataset, m, mode, spec.neer_register_col,
                                    spec.neer_register_value, use_spacy)) is not None]
    if not rows:
        print(f"[entity_analysis] no scored use-case rows found for {spec.display}/{mode}.")
        return
    df = pd.DataFrame(rows).sort_values(["neer_pct", "model"], kind="stable")
    out = analysis_dir(dataset)
    df.to_csv(os.path.join(out, f"entity_neer_{mode}.csv"), index=False)
    md = df[["display", "n_clips", "n_entities", "entity_recall_pct", "neer_pct"]].copy()
    md.columns = ["Model", "Use-case clips", "Entities", "Entity recall %", "NEER %"]
    with open(os.path.join(out, f"entity_neer_{mode}.md"), "w") as f:
        f.write(f"# Named/numeric entity error rate: {spec.display} (use-case register), `{mode}`\n\n")
        f.write("NEER = 1 − (reference entity tokens recovered in the hypothesis / total entity tokens). "
                "Entities = digit-bearing tokens, currency, codes, UPI/emails "
                f"(+ spaCy NER: {'on' if use_spacy else 'off'}).\n\n")
        f.write(build_md_table(md) + "\n")
    print(f"[entity_analysis] {spec.display}/{mode}: wrote entity_neer_{mode}.{{csv,md}}")
    print(md.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="svarah")
    ap.add_argument("--mode", default=PRIMARY_MODE)
    ap.add_argument("--spacy", action="store_true", help="also use spaCy NER (needs en_core_web_sm)")
    a = ap.parse_args()
    main(a.dataset, a.mode, a.spacy)
