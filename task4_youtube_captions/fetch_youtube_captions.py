"""
Stage 1: YouTube Closed Captions Fetch.

Fetches English captions from YouTube for all 986 test IDs.
Video IDs in the TIE_shorts dataset ARE YouTube video IDs (NPTEL lectures).

Saves to results/stage1_raw_transcripts/wer_youtube_raw.csv.
No GPU needed. Run once; re-run normalize_and_score.py for WER.

Caption types (in priority order):
  1. manual  — human-created captions
  2. auto    — YouTube auto-generated captions (Google ASR)

caption_type column values:
  "manual"      — human captions fetched
  "auto"        — auto-generated captions fetched
  "unavailable" — no English captions on this video
  "ip_blocked"  — YouTube blocked requests; saved empty, skip and resume later
  "error"       — other fetch error
"""

import os
import sys
import time
import warnings

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.io_helpers import load_dataset_test, results_dir, stage1_raw_dir

warnings.filterwarnings("ignore")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YTT_AVAILABLE = True
except ImportError:
    print("ERROR: youtube-transcript-api not installed.")
    print("Run: pip install youtube-transcript-api")
    sys.exit(1)

# --------------- Config ---------------
DELAY_BETWEEN_REQUESTS = 1.5   # seconds between requests — avoids rate limiting
IP_BLOCK_WAIT = 60             # seconds to wait when IP blocked before giving up
MAX_IP_BLOCKS = 3              # stop fetching after this many IP blocks in a row
RETRY_ON_ERROR = 2             # retries per video on non-block errors
MODEL_NAME = "youtube"

api = YouTubeTranscriptApi()
consecutive_ip_blocks = 0


def fetch_caption(video_id: str) -> tuple[str, str]:
    """Fetch English captions for a YouTube video ID.

    Returns (text, caption_type).
    """
    global consecutive_ip_blocks

    for attempt in range(RETRY_ON_ERROR + 1):
        try:
            # list() shows available transcripts with type info
            transcript_list = api.list(video_id)

            fetched_text = None
            fetched_kind = None

            # Try manual English first
            try:
                t = transcript_list.find_manually_created_transcript(["en"])
                data = t.fetch()
                fetched_text = " ".join(s.text for s in data).strip()
                fetched_kind = "manual"
            except Exception:
                pass

            # Fall back to auto-generated
            if fetched_text is None:
                try:
                    t = transcript_list.find_generated_transcript(["en"])
                    data = t.fetch()
                    fetched_text = " ".join(s.text for s in data).strip()
                    fetched_kind = "auto"
                except Exception:
                    pass

            if fetched_text is not None:
                consecutive_ip_blocks = 0
                return fetched_text, fetched_kind
            else:
                return "", "unavailable"

        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e)

            # IP blocked
            if "IpBlocked" in err_type or "IpBlocked" in err_msg or "ip_blocked" in err_msg.lower():
                consecutive_ip_blocks += 1
                tqdm.write(f"\n  [IP BLOCKED] #{consecutive_ip_blocks}/{MAX_IP_BLOCKS} — "
                           f"waiting {IP_BLOCK_WAIT}s then {'retrying' if consecutive_ip_blocks < MAX_IP_BLOCKS else 'stopping'}...")
                time.sleep(IP_BLOCK_WAIT)
                if consecutive_ip_blocks >= MAX_IP_BLOCKS:
                    tqdm.write("  [STOPPING] Too many IP blocks. Saving checkpoint and exiting.")
                    return "", "ip_blocked"
                continue

            # Captions disabled or not found
            if any(x in err_type for x in ["TranscriptsDisabled", "NoTranscriptFound", "NoTranscriptAvailable"]):
                consecutive_ip_blocks = 0
                return "", "unavailable"

            # Video unavailable / deleted
            if "VideoUnavailable" in err_type or "NotTranslatable" in err_type:
                consecutive_ip_blocks = 0
                return "", "unavailable"

            # Other error — retry
            if attempt < RETRY_ON_ERROR:
                time.sleep(3)
                continue

            consecutive_ip_blocks = 0
            return "", "error"

    return "", "error"


# --------------- Load dataset ---------------
print("Loading dataset...")
ds = load_dataset_test()

# --------------- Resume from checkpoint ---------------
checkpoint_path = os.path.join(results_dir(), f"wer_{MODEL_NAME}_partial.csv")
completed_ids: set[str] = set()
checkpoint_rows: list[dict] = []

if os.path.exists(checkpoint_path):
    df_partial = pd.read_csv(checkpoint_path)
    completed_ids = set(df_partial["ID"].astype(str).tolist())
    checkpoint_rows = df_partial.to_dict("records")
    print(f"  Resuming from checkpoint: {len(completed_ids)} samples already done\n")

all_rows: list[dict] = []
stats = {"manual": 0, "auto": 0, "unavailable": 0, "ip_blocked": 0, "error": 0}
ip_block_stop = False

print(f"--- Fetching YouTube captions ({len(ds)} samples) ---")
print(f"    Delay: {DELAY_BETWEEN_REQUESTS}s/request  "
      f"IP block limit: {MAX_IP_BLOCKS}  "
      f"Resume support: yes\n")

for sample in tqdm(ds, desc="fetching captions"):
    transcript = (sample.get("Transcript") or "").strip()
    if not transcript:
        continue

    sample_id = sample.get("ID", "")

    # Resume: already have this sample
    if str(sample_id) in completed_ids:
        ckpt_row = next((r for r in checkpoint_rows if str(r["ID"]) == str(sample_id)), None)
        if ckpt_row is not None:
            all_rows.append(ckpt_row)
            continue

    # If IP blocked too many times, stop fetching and mark remaining as ip_blocked
    if ip_block_stop:
        hyp_raw, caption_type = "", "ip_blocked"
    else:
        hyp_raw, caption_type = fetch_caption(sample_id)

        # Check if we hit the IP block limit inside fetch_caption
        if caption_type == "ip_blocked" and consecutive_ip_blocks >= MAX_IP_BLOCKS:
            ip_block_stop = True

    stats[caption_type] = stats.get(caption_type, 0) + 1

    row = {
        "split": "test",
        "ID": sample_id,
        "Speaker_ID": sample.get("Speaker_ID", ""),
        "Gender": sample.get("Gender", ""),
        "Speech_Class": sample.get("Speech_Class", ""),
        "Native_Region": sample.get("Native_Region", ""),
        "Speech_Duration_seconds": sample.get("Speech_Duration_seconds") or "",
        "Discipline_Group": sample.get("Discipline_Group", ""),
        "Topic": sample.get("Topic", ""),
        "transcript_raw": transcript,
        "normalised_transcript_raw": str(sample.get("Normalised_Transcript") or "").strip(),
        "hypothesis_raw": hyp_raw,
        "caption_type": caption_type,
    }

    all_rows.append(row)
    checkpoint_rows.append(row)

    # Save checkpoint every 50 samples
    if len(all_rows) % 50 == 0:
        pd.DataFrame(checkpoint_rows).to_csv(checkpoint_path, index=False)
        tqdm.write(
            f"  [checkpoint] {len(all_rows)} done — "
            f"manual:{stats['manual']} auto:{stats['auto']} "
            f"unavailable:{stats['unavailable']} blocked:{stats['ip_blocked']}"
        )

    if not ip_block_stop:
        time.sleep(DELAY_BETWEEN_REQUESTS)

# --------------- Save ---------------
out_path = os.path.join(stage1_raw_dir(), f"wer_{MODEL_NAME}_raw.csv")
pd.DataFrame(all_rows).to_csv(out_path, index=False)

available = stats["manual"] + stats["auto"]
total = len(all_rows)

print(f"\nSaved: {out_path}")
print(f"Total samples : {total}")
print(f"  manual      : {stats['manual']}")
print(f"  auto        : {stats['auto']}")
print(f"  unavailable : {stats['unavailable']}")
print(f"  ip_blocked  : {stats['ip_blocked']}")
print(f"  error       : {stats['error']}")
print(f"  Coverage    : {available}/{total} ({available/total*100:.1f}%)" if total else "")

if ip_block_stop:
    print(f"\n  [NOTE] Stopped early due to IP blocking after {MAX_IP_BLOCKS} consecutive blocks.")
    print(f"         Re-run this script later to fetch remaining {stats['ip_blocked']} samples.")
    print(f"         Checkpoint saved — already fetched samples will be skipped.")
else:
    if os.path.exists(checkpoint_path):
        os.unlink(checkpoint_path)

print("\nRun 'python normalize_and_score.py' for WER evaluation.")
print("\nDone.")
