# legacy/ — archived data from the first corpus collector

These files are **not used by the current pipeline** and must not be deleted.

## What is here

`dset_raw/` — the shards produced by the first collector (`dset/collector.py`)
on 2026-10-19 … 2026-10-21:

| File | Size | Samples |
|------|------|---------|
| `shard_00001.jsonl` | 8.1 MB | 1015 |
| `shard_00002.jsonl` | 8.1 MB | 1030 |
| `shard_00003.jsonl` | 4.6 MB | 583 |

Total: **2 628 samples**, ~21 MB. The corpus is general programming Q&A with
answers in many languages (Python, C, C++, Rust, Go, JavaScript, …), collected
before the project pivoted to the narrow x86-64 asm → C dataset.

Each line is one JSON record:

`id, ts, model, category, lang, temperature, prompt, response,
prompt_tokens, code_valid, code_error`

## Integrity

`SHA256SUMS` contains checksums of the three shards. Verify with:

```bash
cd legacy
sha256sum -c SHA256SUMS
```

An independent copy (same files and `SHA256SUMS`) lives outside the
repository at `~/Desktop/kind_picture_legacy_dset_raw/`.

## If the data is needed again

Move or copy the shards back to `dset/raw/`:

```bash
mkdir -p dset/raw
mv legacy/dset_raw/*.jsonl dset/raw/
```

`dset/build_dataset.py` expects them there (its default `RAW_DIR` is
`dset/raw`). Keep the Desktop copy untouched as the backup.
