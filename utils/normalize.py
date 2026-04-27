"""Research-grade forward normalization for ASR WER evaluation.

4 evaluation modes — 2 reference sources × 2 normalization states:

    transcript_raw   — Transcript as-is vs Whisper as-is
    transcript_clean — Transcript normalized vs Whisper normalized
    hf_raw           — Normalised_Transcript as-is vs Whisper as-is
    hf_clean         — Normalised_Transcript normalized vs Whisper normalized

All modes are symmetric: same normalization applied to both ref and hyp.
Normalization direction: forward (words/lowercase) — standard for WER.
"""

import re
import unicodedata

try:
    from num2words import num2words as _num2words
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _NUM2WORDS_AVAILABLE = False

MODES = ("transcript_raw", "transcript_clean", "hf_raw", "hf_clean")

_REFERENCE_SOURCE = {
    "transcript_raw":   "Transcript",
    "transcript_clean": "Transcript",
    "hf_raw":           "Normalised_Transcript",
    "hf_clean":         "Normalised_Transcript",
}

_IS_NORMALIZED = {
    "transcript_raw":   False,
    "transcript_clean": True,
    "hf_raw":           False,
    "hf_clean":         True,
}

_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "it's": "it is", "that's": "that is", "what's": "what is",
    "there's": "there is", "here's": "here is", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "i'd": "i would",
    "he's": "he is", "she's": "she is", "they're": "they are",
    "we're": "we are", "you're": "you are", "we've": "we have",
    "they've": "they have", "you've": "you have",
    "could've": "could have", "would've": "would have",
    "should've": "should have", "let's": "let us",
    "that'll": "that will", "who's": "who is",
    "how's": "how is", "where's": "where is",
}

_ORDINAL_PATTERN = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)
_CARDINAL_PATTERN = re.compile(r'\b\d+(\.\d+)?\b')


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return ""
    return str(val)


def _expand_contractions(text: str) -> str:
    for contraction, expansion in _CONTRACTIONS.items():
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        text = pattern.sub(expansion, text)
    return text


def _fix_possessives(text: str) -> str:
    text = re.sub(r"(\w+)'s\b", r"\1 s", text)
    text = re.sub(r"'", "", text)
    return text


def _strip_thousands_separators(text: str) -> str:
    return re.sub(r'(\d),(\d)', r'\1\2', text)


def _ordinal_to_words(text: str) -> str:
    if not _NUM2WORDS_AVAILABLE:
        return text
    def replace_ordinal(m):
        try:
            return _num2words(int(m.group(1)), to="ordinal")
        except Exception:
            return m.group(0)
    return _ORDINAL_PATTERN.sub(replace_ordinal, text)


def _cardinal_to_words(text: str) -> str:
    if not _NUM2WORDS_AVAILABLE:
        return text
    text = _strip_thousands_separators(text)
    def replace_cardinal(m):
        token = m.group(0)
        try:
            if "." in token:
                parts = token.split(".")
                left = _num2words(int(parts[0]))
                right = " ".join(_num2words(int(d)) for d in parts[1])
                return f"{left} point {right}"
            else:
                return _num2words(int(token))
        except Exception:
            return token
    return _CARDINAL_PATTERN.sub(replace_cardinal, text)


def _remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Apply forward normalization: lowercase, expand contractions,
    fix possessives, convert numbers to words, remove punctuation.

    Used for all *_clean modes. Applied symmetrically to both ref and hyp.
    """
    if not text or not text.strip():
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _expand_contractions(text)
    text = _fix_possessives(text)
    text = _ordinal_to_words(text)
    text = _cardinal_to_words(text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = _normalize_whitespace(text)
    return text


def get_reference_source(mode: str) -> str:
    if mode not in _REFERENCE_SOURCE:
        raise ValueError(f"Unknown mode: {mode}. Valid: {MODES}")
    return _REFERENCE_SOURCE[mode]


def get_ref_and_hyp(
    sample: dict,
    hyp_raw: str,
) -> dict[str, dict]:
    """Return a dict of all 4 modes, each containing ref_raw, ref, hyp_raw, hyp.

    Returns:
        {
            "transcript_raw":   {"ref_raw": ..., "ref": ..., "hyp_raw": ..., "hyp": ...},
            "transcript_clean": {...},
            "hf_raw":           {...},
            "hf_clean":         {...},
        }
    """
    transcript = _safe_str(sample.get("Transcript"))
    normalised = _safe_str(sample.get("Normalised_Transcript"))
    hyp_raw_s = _safe_str(hyp_raw)

    hyp_clean = normalize_text(hyp_raw_s) if hyp_raw_s.strip() else ""
    transcript_clean = normalize_text(transcript) if transcript.strip() else ""
    normalised_clean = normalize_text(normalised) if normalised.strip() else ""

    return {
        "transcript_raw": {
            "ref_raw": transcript.strip(),
            "ref":     transcript.strip(),
            "hyp_raw": hyp_raw_s.strip(),
            "hyp":     hyp_raw_s.strip(),
        },
        "transcript_clean": {
            "ref_raw": transcript.strip(),
            "ref":     transcript_clean,
            "hyp_raw": hyp_raw_s.strip(),
            "hyp":     hyp_clean,
        },
        "hf_raw": {
            "ref_raw": normalised.strip(),
            "ref":     normalised.strip(),
            "hyp_raw": hyp_raw_s.strip(),
            "hyp":     hyp_raw_s.strip(),
        },
        "hf_clean": {
            "ref_raw": normalised.strip(),
            "ref":     normalised_clean,
            "hyp_raw": hyp_raw_s.strip(),
            "hyp":     hyp_clean,
        },
    }
