# ml/asm — C → ASM dataset pipeline

Narrow-domain dataset for a small programming model focused on x86-64 assembly
in AT&T syntax (the syntax `gcc -S` emits).

Two separate lines, kept in separate files on purpose:

| Line | What it produces | Verifier | File |
|------|------------------|----------|------|
| A1 | C source → real gcc asm listing (ground truth by construction) | not needed — the compiler made the asm | `gen_c.py` |
| B | task → `_start` program (syscalls only) + test run | `verifier.py` | planned |

## Files

- `gen_c.py` — collector for line A1 (pairs "C source → gcc listing", `-O0` and `-O2`).
- `verifier.py` — runs and checks asm programs written against the contract:
  x86-64, AT&T syntax, entry point `_start`, no libc, I/O only through
  `read(0)` / `write(1)` syscalls. Exit codes: `0` all tests pass, `1` assembled
  but tests failed, `2` assembly/link failure, `3` bad input.
- `tests/` — self-tests for the verifier (`hello.s`, `echo.s`, `crash.s`, `infinite.s`).
- `raw_c/` — output shards of `gen_c.py` (runtime data, not committed).
- `state_gen_c.json` — checkpoint of `gen_c.py` (runtime data, not committed).

## One sample of `gen_c.py`, step by step

1. `index → (task, wording variation)` is deterministic, same scheme as
   `dset/collector.py`: 72 tasks × 10 variations = **720 combinations**.
   Once the space is exhausted the prompts repeat, but the answers differ
   (the temperature is random on every request).
2. A request goes to `/api/chat` (model `beam:cloud`). The system prompt
   demands exactly one ` ```c ` block.
3. The C code is extracted from the response: first a block tagged `c` is
   looked for; if there is none, any block is accepted; if there are several,
   the longest one wins.
4. gcc compiles the code **twice** — with `-O0` and `-O2` — producing two
   listings of the same program. The difference between them (how the
   optimizer rewrites the code) is a valuable part of the dataset in itself.
5. The JSONL record is appended with flush + fsync (it survives a laptop
   shutdown), and the checkpoint `state_gen_c.json` is written atomically
   (temp file + `os.replace` + directory fsync).

### Record fields

`id, ts, model, gcc, task, variation, temperature, prompt, extract_ok,
c_source, prompt_tokens, compile_o0_ok, asm_o0, diag_o0, compile_o2_ok,
asm_o2, diag_o2`

`diag_*` holds gcc warnings/errors: they are data about the quality of the
generated C, not a filter. Records are kept even when gcc fails (the `diag_*`
fields explain why); `compile_*_ok` is the only quality label in line A1.

### Listing cleanup

Compiler service lines (`.file`, `.ident`), header comments (`# GNU C11 …`,
`# options passed …`) and empty linker note sections (`.section .note…`) are
stripped for readability. Everything else is kept, including `# prog.c:7: …`
comments that tie each instruction back to its source line
(`-fverbose-asm`). The gcc version is recorded in the `gcc` field because the
header is removed.

## Usage

```bash
cd ml/asm
python3 gen_c.py --self-test   # offline check of the gcc part, no network
python3 gen_c.py --dry-run 5   # print the next 5 prompts, no network
python3 gen_c.py --status      # show the checkpoint
python3 gen_c.py --once        # one sample, then exit
python3 gen_c.py               # endless collection, Ctrl+C to stop
```

```bash
python3 verifier.py prog.s --tests prog.test.json        # human-readable
python3 verifier.py prog.s --tests prog.test.json --json # machine-readable
```
