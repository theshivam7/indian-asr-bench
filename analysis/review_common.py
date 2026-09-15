"""Constants shared by the human-review scripts (build, fill, stats)."""

# The five systems whose hypotheses appear on every review sheet. The first four
# drive clip selection; Medium is included for context.
REVIEW_MODELS = ("large", "parakeet", "parakeet_ctc", "qwen3", "medium")

REVIEW_FOLDERS = {"tie": "tie_validation", "svarah": "svarah_validation", "aesrc": "aesrc_validation"}

# Closed-set dropdowns: a fixed vocabulary makes the filled sheet tabulable
# without re-normalizing free text first.
CHECK_OPTIONS = ["Correct", "Partially correct", "Incorrect"]
REVIEWER_DECISION_OPTIONS = [
    "Genuine model error", "Reference error", "Audio artifact",
    "Not a real error", "Unsure",
]

# One label vocabulary across all three sheets. Comma separated when a clip has
# several causes.
LABELS = [
    "Reference error", "Misalignment", "Truncated audio", "Disfluency",
    "Number formatting", "Technical vocabulary", "Acronym or code",
    "Hindi named entity", "Indian-language named entity",
    "Foreign named entity", "English name or rare word",
    "Brand or product name", "Accent / pronunciation", "Spelling variant",
    "Short utterance",
]
