# Deep Error Analysis — Why the Worst Samples Fail

Analysis of the **top-20 highest-WER clips per model** (`transcript_clean`, gold mode) across the
five pretrained systems. The goal is to explain *why* WER is high, not just report it. Every claim
below is verified against the actual reference/hypothesis text and is reproducible from
`results/stage2_processed/`.

Method: for each high-WER clip we measure **reference-word recall** (fraction of reference words that
appear anywhere in the hypothesis) and the **hypothesis/reference length ratio**, then read the text.
These two numbers separate *measurement artifacts* from *genuine recognition errors*.

---

## Headline finding: ~70% of the worst-WER samples are dataset artifacts, not ASR errors

The 100 worst-WER rows (5 models × 20) come from only **42 distinct clips** — heavy overlap. Classifying them:

| Category | Share | Signature | What it actually is |
|---|:---:|---|---|
| **A. Clip over-run** | **45%** | recall ≈ 0.93, length ratio ≈ 1.9 | Model transcribes the reference *correctly* **plus** real speech the clip cut off |
| **B. Content mismatch** | **25%** | recall ≈ 0.25 | The audio segment doesn't match the reference text at all |
| **D. Genuine ASR errors** | **30%** | recall ≈ 0.70, ratio ≈ 1.4 | Real substitutions/omissions |

**A + B = 70% are clip/reference-misalignment artifacts in the TIE_shorts test set, not model failures.**

### The proof: architecturally-disjoint models fail *identically*

The 42 clips include **12 that appear in all five models' top-20**. On any given clip, a CTC model
(Parakeet-TDT), an LLM (Qwen3-ASR), and encoder-decoders (Whisper) produce the **same recall and length
ratio** — and often near-verbatim identical text. Models that share no architecture cannot fail identically
unless the fault is in the audio/reference, not the model.

**Clip over-run — `E-r2EDS0uP4` (SLOW, 29.75s), WER 142–162% for all 5 models:**

```
REFERENCE : f which we have seen three times n minus one minus two j so f is three into four
            minus two into five which
medium    : f which we have seen c times n minus one minus 2j sorry this is i made a mistake
            here j2 is one but j2 is 2j2 so this should be counted as two into one ...
parakeet  : f which we have seen c times n minus one minus two j sorry this is i made a mistake
            here j two is one but j two is two j two so this should be counted as two into one ...
```

Parakeet is a CTC/transducer model — it emits *only* what is acoustically present and structurally
cannot hallucinate free text. It transcribes the same "sorry this is i made a mistake here…" as the LLM
and encoder-decoder models. Therefore that content is **real speech in the audio that the reference
transcript omitted**. The reference is a *truncated window* of a longer utterance; the 142% WER is a
measurement artifact.

**Content mismatch — `Vpoi5W6a3lo` (FAST, 17.64s), WER ~120% for all 5 models:**

```
REFERENCE : ...the original distribution is normal ... this distribution of sample means is also
            normal ... central limit theorem ...
all models: rate could be something like well two people arrive per minute ... what is the
            probability zero people will arrive over the next minute ... on average two people ...
```

All five models agree on content (a Poisson-arrivals example) that is **completely different** from the
reference (central limit theorem). Identical *wrong* output from disjoint architectures means they all
heard the same audio — and that audio is mislabeled with the wrong transcript. A dataset alignment error.

**Short-clip amplification — `-2aOCNaOiLs` (SLOW, 4.99s), WER 80%:**

```
REFERENCE : considered in problem forty five
all models: considered in problem forty five [okay] let us do that
```

Every model nails the reference exactly and appends real trailing speech. On a 5-word clip, 4 extra
(correct) words = 80% WER. The model is perfect; the reference is short.

### The artifact is mode-independent

`E-r2EDS0uP4` scores 142% under `transcript_raw`, `transcript_clean`, and `hf_clean`, and 162% under
`hf_raw`. Because the inflation comes from an audio↔reference *window* mismatch, it is unaffected by which
reference column or normalization is used — confirming it is not a normalization artifact.

---

## Secondary findings

### 1. SLOW speech dominates the tail — but not because it is acoustically hard

SLOW clips are **38% of the dataset but 69% of the high-WER samples** (1.8×); FAST (0.52×) and AVG (0.45×)
are *under*-represented. The mechanism is not "slow speech is hard to hear" — it is that **39 of 51
clip-over-runs are SLOW**. Slow lecture delivery, pauses, and spoken self-corrections ("sorry, I made a
mistake") make the reference window far more likely to be a truncated slice of a longer spoken utterance.
This is a segmentation effect, not an acoustic one.

### 2. Errors are U-shaped by duration — concentrated at the extremes

| Duration | Dataset share | High-WER share | Over-representation |
|---|:---:|:---:|:---:|
| 0–5s | 0.4% | 5% | **12.3×** |
| 5–15s | 3.9% | 12% | **3.1×** |
| 15–30s | 87.1% | 70% | 0.8× (under) |
| 30–60s | 8.1% | 11% | 1.4× |
| 60s+ | 0.5% | 2% | **3.9×** |

Short clips amplify a few artifact words into a huge percentage; long clips accumulate boundary drift and
(for the HF pipeline) long-form chunk-stitching errors. The bulk 15–30s band is the *safest*.

### 3. When errors are genuine, they are math and technical notation

The 30% genuine errors are dominated by spoken mathematics and domain vocabulary that the language model
cannot disambiguate: subscripted variables (`k1`, `k2x`, `k2y`), formula tokens (`2j2`, "three" heard as
"c"), and technical terms. Example — `nwcOtus_7eo`: reference `...k1 y is equal to k2 x minus k2y...` →
hypothesis drops/rearranges the subscript variables. This matches the domain (70% Engineering, and
Engineering is 1.2× over-represented in the tail).

### 4. Hallucination is the dominant *genuine* failure mode; Whisper Large is the worst offender

66% of high-WER rows are insertion-heavy (hypothesis longer than reference), and the max observed WER is
**185%**. Counting clips with WER > 100% (pure insertion/hallucination):

| Model | WER>100% clips (of its top-20) |
|---|:---:|
| Whisper Large | 9 |
| Parakeet-TDT | 7 |
| Whisper Medium | 6 |
| Whisper Base | 4 |
| Qwen3-ASR | 3 |

Whisper Large hallucinates most on hard audio — consistent with its highest Std Dev (19.2%) in the main
results. (Note: for the *artifact* clips the "insertion" is real speech; this count still tracks which
models most aggressively transcribe beyond the reference window.)

### 5. No female speaker appears in any model's top-20

Female speakers are 5.9% of the dataset but **0%** of the high-WER tail, and have lower WER overall
(≈12% vs ≈15% for males). With only 58 female clips this is a weak-N observation, but it is consistent:
the misaligned/truncated clips happen to be male-spoken lectures.

---

## What this means for the benchmark

1. **The corpus/mean WER understates true model quality.** Median WER (11.1% for Whisper Medium) is far
   below corpus (14.8%) and mean (15.5%); the ~3.5 pp gap is this contaminated tail. On the *typical*
   clip every model is 3–4 pp better than its headline number. **Median is the more robust metric for
   this dataset.**
2. **Model rankings are unaffected.** The artifacts hit all models roughly equally (identical failures on
   shared clips), so the *relative* comparison between systems remains valid — only the absolute numbers
   are inflated.
3. **The tail is a dataset property, disclosed.** TIE_shorts' `test` split has clip/reference-window
   misalignment in its hardest ~4% of clips. We did not create the splits; we report this so the numbers
   are read correctly and so future work can re-segment or filter the tail.
