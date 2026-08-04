# Resolved environments

Exact package sets captured from the NSCC cluster on 2026-07-30, the last day of access. These
are what the published results were produced with, recovered before the account lapsed.

`environments/parakeet.yaml` and `environments/qwen3.yaml` no longer solve: channel drift made
their MKL / llvm-openmp / mkl_random build hashes mutually unsatisfiable, and conda's solver
fails rather than substituting. These files are the working alternative.

| File | Source | Use |
|---|---|---|
| `<env>.explicit.txt` | `conda list --explicit` | Exact builds and package URLs. Resolves without a solver |
| `<env>.pipfreeze.txt` | `pip freeze` | The pip side, which is most of `parakeet` and `qwen3` |

## Rebuild

```bash
conda create -n whisper           --file environments/resolved/whisper.explicit.txt
conda create -n whisper_medium_ft --file environments/resolved/whisper_medium_ft.explicit.txt

# parakeet and qwen3 were built from pip on a bare interpreter, so rebuild them that way
conda create -n parakeet python=3.10 -y
conda run -n parakeet pip install -r environments/resolved/parakeet.pipfreeze.txt
conda create -n qwen3 python=3.10 -y
conda run -n qwen3 pip install -r environments/resolved/qwen3.pipfreeze.txt
```

## Two things to know

**The GPU build must stay pinned.** A pip-installed torch does not reliably see the GPU on an
A100 cluster; the conda build string is what makes it work:

```
pytorch-2.5.1-py3.10_cuda11.8_cudnn9.1.0_0
pytorch-cuda-11.8-h7e8668a_6
```

**These envs drifted after the runs.** `parakeet.pipfreeze.txt` shows `nemo-toolkit==2.3.0` and
`transformers==5.14.1`, while the run manifests in
`results/*/stage1_raw_transcripts/*_manifest.json` record `nemo_toolkit 2.7.3` and
`transformers 4.46.3` for the runs that produced the committed transcripts. Use the manifests to
know what produced a given result, and these files to get a working environment. They answer
different questions.

Only Stage 1 needs any of this. Stage 2 and 3 reproduce every table and figure from the committed
transcripts with the top-level `requirements.txt` alone.
