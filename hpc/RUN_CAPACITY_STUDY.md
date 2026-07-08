# Running the Tiny/Small capacity study on NSCC

I can't reach NSCC from this session (the jump host is a private NTU IP, non-key auth) —
run these yourself from the login node (`asp2a-login-ntu01`). Everything below is already
committed to the repo; `git pull` first.

## 0. Environment (once, if not already set)

```bash
export WORKDIR=/path/to/indian-asr-bench   # your repo clone on NSCC
export HF_CACHE=/scratch/$USER/hf_cache
export CONDA_BASE=$(conda info --base)
export CUDA_MODULE=cuda/11.8.0
export WHISPER_ENV=whisper WHISPER_FT_ENV=whisper_medium_ft
cd "$WORKDIR" && git pull
```

## 1. Smoke test (do this first, on the login node, CPU is fine)

Validates `finetune_stepwise.py`'s TIE data path + filters end-to-end before burning a GPU
allocation. TIE's train split should already be in `$HF_CACHE` from prior runs.

```bash
conda activate whisper_medium_ft
HF_DATASETS_CACHE=$HF_CACHE HF_HOME=$HF_CACHE \
python task6_whisper_medium_ft/finetune_stepwise.py \
    --base-model openai/whisper-tiny \
    --output-dir /tmp/smoke_tiny_ft \
    --max-steps 20 --max-train-samples 40
```

Pass criteria: prints train/validation filter counts (should roughly match finetune.py's
historical ~7,200/986, possibly a handful fewer — bug-fix (a), see the findings report),
trains 20 steps without error, prints an `eval_wer`, and writes `config.json` +
`model.safetensors` + `preprocessor_config.json` etc. to `/tmp/smoke_tiny_ft`. If it fails,
stop and report the traceback rather than submitting the real jobs.

Also smoke-test the analysis refactor (no GPU needed, in your local analysis env):

```bash
python analysis/compare_finetune.py    # should reproduce committed medium outputs
git diff --stat results/tie/analysis/  # only finetune_comparison.md text + new files should differ
```

## 2. Submit the real jobs (serially — do not run in parallel)

```bash
cd "$WORKDIR"

# A. Pretrained tiny + small on TIE (openai-whisper engine)
JOB_A=$(qsub -P $PROJECT -v "WORKDIR=$WORKDIR,HF_CACHE=$HF_CACHE,CONDA_BASE=$CONDA_BASE,CUDA_MODULE=$CUDA_MODULE,WHISPER_ENV=$WHISPER_ENV,MODEL=tiny,DATASET=tie" hpc/job_whisper.pbs)
JOB_B=$(qsub -P $PROJECT -W depend=afterany:$JOB_A -v "WORKDIR=$WORKDIR,HF_CACHE=$HF_CACHE,CONDA_BASE=$CONDA_BASE,CUDA_MODULE=$CUDA_MODULE,WHISPER_ENV=$WHISPER_ENV,MODEL=small,DATASET=tie" hpc/job_whisper.pbs)

# C. Fine-tune tiny (+ its two HF-pipeline evals)
JOB_C=$(qsub -P $PROJECT -W depend=afterany:$JOB_B -v "WORKDIR=$WORKDIR,HF_CACHE=$HF_CACHE,CONDA_BASE=$CONDA_BASE,CUDA_MODULE=$CUDA_MODULE,WHISPER_FT_ENV=$WHISPER_FT_ENV,SIZE=tiny" hpc/job_finetune_size.pbs)

# D. Fine-tune small (+ its two HF-pipeline evals)
JOB_D=$(qsub -P $PROJECT -W depend=afterany:$JOB_C -v "WORKDIR=$WORKDIR,HF_CACHE=$HF_CACHE,CONDA_BASE=$CONDA_BASE,CUDA_MODULE=$CUDA_MODULE,WHISPER_FT_ENV=$WHISPER_FT_ENV,SIZE=small" hpc/job_finetune_size.pbs)

echo "A=$JOB_A B=$JOB_B C=$JOB_C D=$JOB_D"
qstat -u $USER
```

Rough walltimes: A ~20-40min, B ~45-90min, C ~1.5-2.5h (2000 steps + 2 evals), D ~2.5-4h.
Job C/D each already run BOTH the fine-tuned and pretrained-HF-baseline eval internally —
no separate submission needed for `tiny_hf`/`small_hf`.

## 3. Contingency check (after C and after D)

```bash
cat models/whisper_tiny_ft/trainer_state.json | python -c "
import json,sys
s = json.load(sys.stdin)
best = s.get('best_model_checkpoint','?')
print('best checkpoint:', best)
print('log history (step, eval_wer):')
for e in s['log_history']:
    if 'eval_wer' in e: print(' ', e['step'], e['eval_wer'])
"
```

If the best checkpoint is at step ≤200 (out of 2000) and eval_wer never improves after —
that's the "epoch-1 signature" from the medium study. Run ONE diagnostic refit at
`--lr 1e-4` into a **fresh** output dir before concluding anything (auto-resume would
otherwise silently reuse the old LR schedule — there is no auto-resume in
`finetune_stepwise.py` currently, but keep the dirs separate regardless for clarity):

```bash
python task6_whisper_medium_ft/finetune_stepwise.py \
    --base-model openai/whisper-tiny --lr 1e-4 \
    --output-dir models/whisper_tiny_ft_lr1e4
```

The 1e-5 run stays the headline regardless of this outcome — see
`results/tie/analysis/findings_tiny_small_ft.md` (Step 8) for how to report it.

## 4. Push raw results back

```bash
git add results/tie/stage1_raw_transcripts/wer_{tiny,small,tiny_hf,tiny_ft,small_hf,small_ft}_raw.csv \
        results/tie/stage1_raw_transcripts/wer_{tiny,small,tiny_hf,tiny_ft,small_hf,small_ft}_manifest.json
git add models/whisper_tiny_ft/eval_results.json models/whisper_small_ft/eval_results.json \
        models/whisper_tiny_ft/trainer_state.json models/whisper_small_ft/trainer_state.json 2>/dev/null
git commit -m "Add tiny/small pretrained + fine-tuned raw transcripts (capacity study)"
git push
```

Then pull locally and I'll run Stage 2/3 scoring + write the findings report.
