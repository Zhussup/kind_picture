# kind_picture

A learning project on the way to owning a small domain-specific language
model: collect data, build datasets, fine-tune a small pretrained model.

The repository currently contains two independent data pipelines:

| Pipeline | Purpose | Folder |
|----------|---------|--------|
| General programming corpus | broad Q&A/code corpus from an Ollama Cloud model, used as SFT material | `dset/` |
| x86-64 asm → C dataset | narrow dataset for assembly reconstruction with mechanical verification | `ml/asm/` |

```
kind_picture/
├── dset/      corpus collector + dataset builder (general programming)
├── ml/asm/    asm → C dataset pipeline (collector, builder, differential tester, verifier)
├── legacy/    archived shards from the first corpus collector (do not delete)
├── docs/      notes
└── .venv/     local Python environment
```

Requirements: a running Ollama daemon with a logged-in Ollama Cloud account
(the `beam:cloud` model), Python 3 with `requests`, and `gcc`/`binutils` for
the asm pipeline.

---

## dset/ — general programming corpus

The collector runs the free cloud model `beam:cloud` through the Ollama API
around the clock and stores "prompt → answer" dialogues about programming.
`build_dataset.py` later turns them into a fine-tuning dataset.

### Files

| File/folder | What it is |
|---|---|
| `collector.py` | the collector itself (endless loop) |
| `build_dataset.py` | turns collected data into `dataset.jsonl` for fine-tuning |
| `run.sh` | starts it in the background with auto-restart |
| `stop.sh` | stops it |
| `raw/` | data in shards: `shard_00001.jsonl`, `shard_00002.jsonl`, … |
| `state.json` | **checkpoint** — saved progress, so a restart continues where it stopped |
| `collector.log` | work log (every sample and every error) |
| `collector_console.log` | full `run.sh` output (mostly duplicates the log) |
| `dataset.jsonl` | the finished dataset (appears after `build_dataset.py`) |

### First run (sanity check)

1. Ollama must be running and logged in to Ollama Cloud (`ollama run
   beam:cloud` once is enough; the daemon usually stays on port 11434).
2. Check connectivity with a single request:

   ```bash
   cd dset
   python3 collector.py --once
   ```

   It prints `#0 OK ...` and creates `raw/shard_00001.jsonl`.

### Continuous run (24/7)

```bash
cd dset
bash run.sh
```

The script starts the collector in the background via **nohup** (the process
survives closing the terminal) and restarts it in a loop if it ever dies.

### Laptop shutdown / lost network is not a problem

- **No network** — the program does not crash: errors are logged and retried
  with exponential **backoff** (5 s → 10 s → 20 s … up to 1 hour); when the
  network returns, collection resumes.
- **Laptop powered off** — no matter at which moment: every record is
  **flushed + fsync'ed** to disk before moving on (so only complete lines are
  ever on disk), and the checkpoint is written **atomically** (temp file +
  rename — the file is either fully old or fully new, never half-written).

### Restarting (resuming)

```bash
cd dset
bash run.sh
```

Nothing else to pass — the program reads `state.json` and continues from the
last sample. Prompts are generated deterministically from the sample index (no
randomness in prompt construction), so a restart continues exactly at the next
unseen prompt.

Useful commands:

```bash
python3 collector.py --status     # progress (samples_ok, last error, timestamps)
python3 collector.py --dry-run 3  # print the next 3 prompts, no network
tail -f collector.log             # live log
wc -l raw/*.jsonl                 # how many samples collected
```

### Stopping

```bash
cd dset
bash stop.sh
```

### Building the dataset

```bash
cd dset
python3 build_dataset.py
```

This produces `dataset.jsonl` — **JSONL** (one JSON object per line):

```json
{"id": 0, "category": "implement", "lang": "Python",
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

That is the standard chat-dataset format; it works for fine-tuning a small
Qwen-class model (0.5–0.6B) with unsloth / llama-factory / axolotl and is easy
to convert to plain text if needed.

### Built-in protections in `collector.py`

- **Jitter** — a random 0–1.5 s is added to the base 2 s pause, so the traffic
  does not look perfectly machine-timed and is less likely to trip anti-abuse
  protection.
- **Random temperature** (0.6–0.95 per request, stored in the record) — with
  thousands of generations at one temperature the model starts repeating
  itself (same variable names and tricks); this is called **mode collapse**.
- **Syntax check** — every ```python block in an answer is run through
  `ast.parse` (the module that builds a syntax tree; if it fails to parse, the
  code is definitely broken). The result is stored in `code_valid`. Such
  samples are excluded from the dataset by default; bring them back with
  `--include-invalid`.
- **429 / 503** — on "Too Many Requests" the `Retry-After` header (how many
  seconds the server asks to wait) sets a minimum for the backoff
  (5 s → 10 s → … → 1 hour).

### Glossary

- **Ollama API** — HTTP interface of the local ollama server at
  `http://localhost:11434`; the cloud model `beam:cloud` is called the same
  way as a local one (beam is free / zero-rated).
- **Checkpoint** (`state.json`) — a small note "where we stopped": next prompt
  index, shard number, error statistics.
- **JSONL** — a text file where each line is a separate JSON object; easy to
  append one record at a time.
- **Shard** — a piece of data (a file with ~8 MB of answers), so a failure
  damages at most one piece.
- **fsync** — forces the OS to write data to disk immediately instead of
  keeping it in the page cache; otherwise a power loss could lose the last
  minutes of records.
- **Atomic write** — write to a temp file, then rename; rename is atomic, so
  the file can never be half-written.
- **Backoff** — the pause grows on repeated errors (5 s, 10 s, 20 s …) instead
  of hammering a failing server every second.
- **nohup** — makes a process survive closing the terminal.

---

## ml/asm/ — x86-64 asm → C dataset

The narrow pipeline: the model receives an x86-64 assembly listing in AT&T
syntax produced by gcc and must reconstruct C with the same observable
behavior. Correctness is checked by differential testing (same stdin → same
stdout/exit as the original), not by comparing C text.

| File | What it does |
|---|---|
| `gen_c.py` | asks `beam:cloud` for a C program, compiles it with gcc at `-O0` and `-O2`, stores both listings |
| `build_dataset.py` | builds chat-format train/val JSONL (listing → original C), strips source-line leaks, deduplicates |
| `difftest.py` | differential tester: compiles reference and predicted C, runs both on identical inputs, compares behavior |
| `verifier.py` | runner/checker for freestanding `_start` asm programs (future line B) |

```bash
cd ml/asm
python3 gen_c.py --once          # one collected sample
python3 gen_c.py                 # endless collection
python3 build_dataset.py         # rebuild dataset/train.jsonl + val.jsonl
python3 difftest.py ref.c cand.c # check a reconstruction
```

Details, contracts, data format and exit codes: [ml/asm/README.md](ml/asm/README.md).

---

## Legacy data

The first corpus (2 628 samples of general programming Q&A in many languages,
~21 MB) was moved to `legacy/dset_raw/` after the project pivoted to the
asm → C dataset. It is not used by the current pipeline and must not be
deleted. Checksums and restore instructions: [legacy/README.md](legacy/README.md).
