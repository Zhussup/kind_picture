<div align="center">

<img src="docs/logo.jpg" alt="kind_picture logo: asm to C in braces" width="150">

# kind_picture

### Teaching a 0.6B model to read x86-64 assembly and write back working C

![task](https://img.shields.io/badge/task-x86--64%20asm%20%E2%86%92%20C-2a78d6?style=flat-square)
![model](https://img.shields.io/badge/model-Qwen3--0.6B-6250d6?style=flat-square)
![gpu](https://img.shields.io/badge/GPU-RTX%203050%206%20GB-1baf7a?style=flat-square)
![status](https://img.shields.io/badge/status-early%20prototype-eb6834?style=flat-square)

**A narrow, mechanically verifiable code model that runs on a laptop.**

[Pipeline](#pipeline) · [Results](#results) · [Quick start](#quick-start) · [Roadmap](#roadmap) · [Docs](#docs)

</div>

---

> **Status (2026-10-22):** the pipeline works end-to-end and the second full
> LoRA run is evaluated. On a 118-pair validation split the adapter compiles
> **70.3%** of the answers and reproduces the reference behavior on **22.9%**;
> on the 26-pair subset shared with the earlier runs it reaches **84.6%** and
> **23.1%**, against 73.1% / 0% for the previous adapter and 26.9% / 0% for the
> base model. Behavior match is no longer zero — the remaining gap is edge-case
> handling, not syntax. Numbers are reported as they are.

## The idea

Most code models are trained on the easy direction: source code → assembly.
Reverse engineering needs the opposite. Given a compiler listing, recover C
that **behaves the same** — and prove it mechanically.

That constraint is what makes this project interesting:

- **Ground truth by construction.** A hosted C program is compiled by gcc at
  `-O0` and `-O2`; the listings are the input, the original C is the target.
  No human labels, no synthetic guesses.
- **Correctness is behavioral, not textual.** The predicted C and the
  reference C are compiled with the same gcc and run on identical stdin. Same
  stdout and exit code on every probe means the reconstruction is correct —
  even if the code looks completely different.
- **Leak-free by design.** `-fverbose-asm` embeds the original source lines in
  comments. The dataset builder removes exactly those lines, so the model
  cannot win by copying.
- **Small enough to own.** Qwen3-0.6B plus a 10.1M-parameter LoRA adapter fits
  a 6 GB laptop GPU. The goal is a domain model you can run, inspect and
  retrain yourself — not an API bill.

## Pipeline

```mermaid
flowchart LR
  T["406 task templates x 32 variations"] --> G["beam:cloud writes C"]
  G --> GCC["gcc -O0 / -O2"]
  GCC --> L["AT&T listing + reference C"]
  L --> B["build_dataset.py<br/>strip leaks, dedupe, hash split"]
  B --> D["train.jsonl / val.jsonl"]
  D --> S["LoRA SFT, Qwen3-0.6B"]
  S --> M["reconstructed C"]
  R["reference C"] --> DT["difftest.py"]
  M --> DT
  DT --> V["compile_rate / pass_rate"]
```

The two data lines are deliberately kept apart:

| Line | Input | Correctness established by | Status |
|------|-------|----------------------------|--------|
| **A1** | gcc-generated hosted C listings (`main`, libc) | ground truth by construction | collected, trained, evaluated |
| **B** | freestanding `_start` programs, syscalls only | `verifier.py` test runs | planned |

Mixing them would create contradictory answers for the same question: hosted
code uses libc and PLT calls, freestanding code talks to the kernel directly.

## Results

| Run | Validation | Compiles | Same behavior | What it shows |
|-----|-----------:|---------:|--------------:|---------------|
| Base Qwen3-0.6B | 26 pairs | 26.9% | 0% | the starting point: often copies the listing into a `c` block |
| Smoke LoRA, 60 pairs x 3 epochs | 16 pairs* | 37.5% | 0% | format learned: **16/16** answers are C; expected to overfit |
| Full LoRA, 723 pairs x 2 epochs | 26 pairs | **73.1%** | 0% | structure learned: all 26 answers are C; behavior is the remaining gap |
| Full LoRA, 2,579 pairs x 3 epochs | 26 pairs | **84.6%** | **23.1%** | same 26 pairs as the rows above: compiles more, and behavior finally matches |
| Full LoRA, 2,579 pairs x 3 epochs | 118 pairs | 70.3% | **22.9%** | the full split; see the figure below |

\* The smoke run used the earlier 16-pair split, so it is not directly
comparable with the 26-pair baseline. Its purpose was to validate the training
loop, not to claim accuracy.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ml/asm/docs/eval_full118_dark.png">
  <img src="ml/asm/docs/eval_full118_light.png"
       alt="Full 118-pair validation split. Compile rate 70.3% overall, 64.4% at -O0 and 76.3% at -O2. Same behavior on all probes 22.9% overall, 23.7% at -O0 and 22.0% at -O2. Of the 118 answers, 27 passed, 56 compiled but differed, 27 did not compile and 8 had no c block."
       width="840">
</picture>

The second full run used every pair that fit the 2,048-token window: 2,579
pairs, 3 epochs, final loss 0.015. The 26-pair row is the subset that the
earlier runs were evaluated on, so it is the only apples-to-apples comparison;
the 118-pair row is the honest number on the whole split and is lower because
that subset happens to be easier than average.

On the full split the adapter compiles **70.3%** of the answers and matches the
reference behavior on **22.9%**. The `-O2` side compiles markedly more often
than `-O0` (76.3% vs 64.4%), but behavior match is flat across the two (22.0%
vs 23.7%): optimizer output is not harder to read, it just fails differently.

Of the 118 answers, 27 match the reference on every probe, 56 compile but
disagree with it, 27 do not compile, and 8 contain no `` ```c `` fence at all.
Sixteen of the 56 mismatches already agree on 9 of the 10 probes, so most of the
remaining gap is edge-case handling (empty input, negative numbers) rather than
the algorithm itself.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="ml/asm/docs/training_loss_full_dark.png">
  <img src="ml/asm/docs/training_loss_full_light.png"
       alt="Full-run training loss on assistant tokens: 0.81 at the first step, 0.06 at the last, over 182 optimizer steps"
       width="840">
</picture>

Cross-entropy on assistant tokens only: the system prompt, the question and
the assembly listing are masked with `-100`, so every gradient comes from the
C answer. In the first run loss fell from 0.81 to 0.06 over 182 optimizer steps
(epoch averages: 0.32 and 0.14); the second run, with three times the data,
went from 0.60 at the first step to 0.015 at the last over 969 steps (epoch
averages: 0.229, 0.103, 0.057).

## Dataset at a glance

| Metric | Value |
|--------|-------|
| Programs collected | **4,901** (collector stopped 2026-10-22) |
| Programs in the last dataset build | 2,906 |
| Listings (`-O0` + `-O2`) in that build | **5,467** |
| Train / validation pairs | **5,349 / 118**, split by hashed source id |
| Prompt space | 406 tasks x 32 wording variations = **12,992** |
| Duplicates removed / compile failures excluded | 237 / 108 |
| Pairs dropped from the last training run as too long | 2,770 of 5,349 (> 2,048 tokens) |
| A rebuild over all collected programs would give | 9,066 / 182 pairs (not yet trained on) |

Every pair carries two views of the same program: `-O0` preserves the source
structure, `-O2` shows what the optimizer does to it. The split is by source
record, so both optimization levels of one program always land on the same
side.

## Quick start

Requirements: Python 3.13, `gcc`/`binutils`, and for training a CUDA GPU
(the current runs use an RTX 3050 Laptop with 6 GB). Versions used here:
PyTorch 2.6.0+cu124, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0,
Matplotlib 3.11.2 (for the training-loss figures).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft accelerate matplotlib requests
```

Build the dataset from the collected shards, train, and evaluate:

```bash
# 1. rebuild train/val from ml/asm/raw_c/ (safe while the collector runs)
python3 ml/asm/build_dataset.py

# 2. smoke test the training loop on 60 short pairs
python3 ml/asm/train.py --limit 60 --epochs 3 --max-seq-len 1024 \
    --grad-accum 4 --tag smoke

# 3. full run (the numbers in Results, ~2h15m on an RTX 3050)
python3 ml/asm/train.py --epochs 3 --max-seq-len 2048 \
    --grad-accum 8 --tag night-20261022

# 4. evaluate on the validation split (`--val` can point at any JSONL of pairs,
#    which is how the same 26-pair subset is compared across runs)
python3 ml/asm/eval.py --model Qwen/Qwen3-0.6B --tag base
python3 ml/asm/eval.py --model Qwen/Qwen3-0.6B \
    --adapter ml/asm/checkpoints/lora-night-20261022/final --tag night-final
python3 ml/asm/eval.py --model Qwen/Qwen3-0.6B \
    --adapter ml/asm/checkpoints/lora-night-20261022/final \
    --val /tmp/subset.jsonl --tag night-final-subset

# 5. regenerate a training-loss figure (light + dark PNG, matplotlib)
python3 ml/asm/plot_training.py \
    --state ml/asm/checkpoints/lora-night-20261022/train_state.json \
    --title "Second run: loss on assistant tokens" \
    --subtitle "2,579 pairs | 3 epochs | 969 optimizer steps" \
    --stem training_loss_night
```

Collect more programs (needs a logged-in Ollama Cloud account and the
`beam:cloud` model):

```bash
cd ml/asm
python3 gen_c.py --once      # one sample
python3 gen_c.py             # endless, crash-safe collection
python3 gen_c.py --status    # checkpoint and error statistics
```

## Repository layout

```
kind_picture/
├── ml/asm/                 asm → C pipeline (the main project)
│   ├── gen_c.py            collector: task → C → gcc -O0/-O2 listings
│   ├── build_dataset.py    leak stripping, dedupe, train/val split
│   ├── difftest.py         behavioral equivalence checker
│   ├── eval.py             compile_rate / pass_rate on validation pairs
│   ├── train.py            LoRA SFT with masked loss
│   ├── plot_training.py    light/dark training-loss figures
│   ├── verifier.py         freestanding asm runner (line B)
│   ├── raw_c/              collected shards, JSONL per 500 programs (not in git)
│   ├── dataset/            generated train.jsonl / val.jsonl (not in git)
│   ├── checkpoints/        LoRA adapters per epoch plus `final` (not in git)
│   ├── eval_results/       per-pair predictions and summary JSON
│   └── docs/               figures used by the READMEs
├── dset/                   earlier general-programming corpus collector
├── legacy/                 archived first corpus, with SHA-256 checksums
├── docs/                   notes
└── README.md
```

## Roadmap

- [x] Crash-safe collector with atomic checkpoints and backoff
- [x] Dataset builder with source-line leak stripping and deduplication
- [x] Differential tester with timeouts, memory limits and signal handling
- [x] Evaluation harness: `compile_rate`, `pass_rate`, per-optimization split
- [x] LoRA training with assistant-only loss and memory-aware logits
- [x] First full training run: 73.1% compile rate on unseen pairs
- [x] Full run with a non-zero `pass_rate` on unseen pairs: 22.9% of 118
- [ ] Per-task differential inputs, so `pass_rate` is not measured on one
      fixed set of 10 probes for every program
- [ ] Line B: freestanding `_start` programs verified by `verifier.py`
- [ ] A larger model on a two-GPU server
- [ ] Inference CLI: listing in, verifiable C out

## Limitations

This is an early research prototype, and the README says so on purpose:

- The task is narrow: small hosted C programs generated from 406 templates,
  compiled by gcc. It is not a general-purpose decompiler.
- The current model is 0.6B parameters. The second full run puts 70.3% of
  unseen answers through the compiler and 22.9% through the differential test;
  the rest either fail to compile or disagree with the reference on at least
  one probe.
- `pass_rate` is measured on one fixed set of 10 stdin probes for every pair,
  not on inputs chosen for the task. It underestimates tasks that need specific
  input and overestimates tasks whose logic those probes never reach.
- Training skips pairs longer than `--max-seq-len` (2,048 tokens in the current
  run) rather than truncating them: 2,770 of 5,349 pairs never reached the last
  run. Long programs are underrepresented.
- The corpus comes from a single cloud model, so its style and mistakes are
  baked into the data.
- Line B (freestanding assembly) is designed but not collected yet.

## Docs

- [ml/asm/README.md](ml/asm/README.md) — full pipeline reference: record
  format, leak stripping, differential-testing rules, exit codes, training
  details.
- [dset/README.md](dset/README.md) — the earlier general-programming corpus
  collector.
- [legacy/README.md](legacy/README.md) — archived first corpus, checksums and
  restore instructions.

<div align="center">

Built on **Qwen3-0.6B** with PyTorch, Transformers and PEFT.

</div>
