"""Research-grade forward normalization for ASR WER evaluation.

Each evaluation *mode* pairs a reference source with a normalizer; the mode
registry itself lives in ``utils.registry`` (single source of truth). This module
implements the three normalizers and a ``normalize_for_mode`` dispatcher.

Normalizers (all applied SYMMETRICALLY to both reference and hypothesis):
    minimal  (minimal_clean_text)  — strip wrapping quotes + lowercase + remove punctuation.
    custom   (normalize_text)      — minimal + possessive fix + number-to-words (project gold).
    whisper  (whisper_normalize_text) — the community-standard Whisper EnglishTextNormalizer,
                                        added for cross-paper comparability.

Reference-source roles ("gold"/"alt") are resolved by the caller via
``utils.registry.get_reference_role`` onto the raw-CSV columns transcript_raw /
normalised_transcript_raw.
"""

import re
import unicodedata

from utils.registry import (
    ALL_MODES as MODES,
    get_normalizer,
    get_reference_role,
)

try:
    from num2words import num2words as _num2words
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _NUM2WORDS_AVAILABLE = False

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
    """Apply forward normalization: fix possessives, convert numbers to words,
    lowercase, remove punctuation, collapse whitespace.

    Contractions are deliberately NOT expanded to their two-word form: instead the
    apostrophe is stripped, so "don't" -> "dont" and "bernoulli's" -> "bernoulli s"
    (see _fix_possessives). This is applied symmetrically to both reference and
    hypothesis, so it never rewards a rewrite that neither transcript uses. Used for
    the *_clean (custom) modes.
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


# --------------------------------------------------------------------------- #
# Whisper EnglishTextNormalizer (community standard) — lazy, cached.
# Packaged by the `whisper_normalizer` PyPI wheel (no torch dependency), so it is
# importable in the CPU-only analysis env. Only required when a `whisper` mode runs.
# --------------------------------------------------------------------------- #
_WHISPER_NORMALIZER = None


def _get_whisper_normalizer():
    global _WHISPER_NORMALIZER
    if _WHISPER_NORMALIZER is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'whisper_norm' mode needs the `whisper_normalizer` package "
                "(pip install whisper_normalizer). It packages Whisper's "
                "EnglishTextNormalizer without pulling in torch."
            ) from e
        _WHISPER_NORMALIZER = EnglishTextNormalizer()
    return _WHISPER_NORMALIZER


def whisper_normalize_text(text: str) -> str:
    """Community-standard Whisper English normalization (Radford et al. 2023).

    Expands contractions, spells out numbers, standardises spelling and strips
    punctuation. Applied symmetrically; used for the `whisper_norm` comparison mode.
    """
    if not text or not text.strip():
        return ""
    return _get_whisper_normalizer()(text).strip()


_NORMALIZERS = {
    "minimal": minimal_clean_text,
    "custom": normalize_text,
    "whisper": whisper_normalize_text,
}


def normalize_for_mode(mode: str, text: str) -> str:
    """Apply the normalizer that `mode` selects (via the registry) to `text`."""
    return _NORMALIZERS[get_normalizer(mode)](text)


def get_reference_source(mode: str) -> str:
    """Backward-compatible alias: returns the canonical reference role (gold/alt)."""
    return get_reference_role(mode)


