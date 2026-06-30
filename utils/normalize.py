"""Research-grade forward normalization for ASR WER evaluation.

4 evaluation modes — 2 reference sources × 2 cleanup levels:

    transcript_raw   — Transcript,            minimal cleanup vs Whisper minimal cleanup
    transcript_clean — Transcript,            full normalization vs Whisper full normalization
    hf_raw           — Normalised_Transcript, minimal cleanup vs Whisper minimal cleanup
    hf_clean         — Normalised_Transcript, full normalization vs Whisper full normalization

Minimal cleanup (minimal_clean_text): strip wrapping quotes + lowercase + remove punctuation.
Full normalization (normalize_text): minimal + possessive fix + number-to-words.
All modes are symmetric: the same cleanup is applied to both ref and hyp.
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

_ORDINAL_PATTERN = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)
_CARDINAL_PATTERN = re.compile(r'\b\d+(\.\d+)?\b')


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return ""
    return str(val)


def strip_wrapping_quotes(text) -> str:
    """Strip a single leading/trailing pair of double quotes and surrounding whitespace.

    The TIE_shorts `Transcript` field often wraps the whole sentence in double quotes,
    e.g. "The second component ..." -> The second component ...
    Only an outer matched pair is removed; quotes inside the sentence are untouched here.
    """
    s = (text or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


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
    """Apply forward normalization: lowercase, fix possessives, convert numbers
    to words, remove punctuation.

    Contractions are intentionally left unexpanded (e.g. "don't" stays "don't"
    rather than becoming "do not"). Used for all *_clean modes. Applied
    symmetrically to both ref and hyp.
    """
    if not text or not text.strip():
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _fix_possessives(text)
    text = _ordinal_to_words(text)
    text = _cardinal_to_words(text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = _normalize_whitespace(text)
    return text


def minimal_clean_text(text: str) -> str:
    """Light cleanup for the *_raw modes: strip wrapping quotes, lowercase,
    remove punctuation, normalize whitespace.

    Deliberately does NOT do number-to-words, possessive splitting, or any other
    full-normalization step. Applied symmetrically to both ref and hyp.
        "The second component is less than here ..."
    -> the second component is less than here ...
    """
    if not text or not text.strip():
        return ""

    text = unicodedata.normalize("NFC", text)
    text = strip_wrapping_quotes(text)
    text = text.lower()
    text = _remove_punctuation(text)
    text = _normalize_whitespace(text)
    return text


def get_reference_source(mode: str) -> str:
    if mode not in _REFERENCE_SOURCE:
        raise ValueError(f"Unknown mode: {mode}. Valid: {MODES}")
    return _REFERENCE_SOURCE[mode]


