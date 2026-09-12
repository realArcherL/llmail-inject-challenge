# LLMail-Inject on Phi-3: measuring prompt-injection defenses

Replays real indirect prompt-injection attacks from Microsoft's
[LLMail-Inject challenge](https://arxiv.org/abs/2506.09956) against
`microsoft/Phi-3-medium-128k-instruct` on rented GPUs, and measures what each defense blocks and
what it costs in answer quality.

Every number in `results/` is produced by a script in `analysis/`, and every raw input and output is
pinned by SHA-256 in that experiment's `manifest.json`.

## Results

| Experiment | Question | Report |
|---|---|---|
| 01 | Does this setup reproduce the challenge's recorded outcomes? | [results/01-reproduction](results/01-reproduction/REPORT.md) |
| 02 | Do Microsoft's spotlighting and the `spotlighting-datamarking` library block the attacks, and at what cost? | [results/02-library-defenses](results/02-library-defenses/REPORT.md) |
| 03 | With no defense at all, what does the model have in mind while it reads an attack? | [results/03-jacobian-lens](results/03-jacobian-lens/REPORT.md) |
| 03, defenses | Does a defense change what the model has in mind, or only what it says? | [REPORT-defenses.md](results/03-jacobian-lens/REPORT-defenses.md) |
| 03, told | What does the model make of being told it was prompt-injected? | [REPORT-told.md](results/03-jacobian-lens/REPORT-told.md) |

## What you need

- **Python 3.10+**: `pip install -r analysis/requirements.txt`
- **Node 20+**: `npm install`
- **A Modal account** for the GPUs, and a spend limit above the costs below
- **Disk**: about 2 GB for the dataset locally, 28 GB on a Modal volume for the model

## Costs, measured

| Step | Answers generated | Cost |
|---|---|---|
| Experiment 01 | 4,800 | about $15 |
| Experiment 02 | 7,264 | about $17 |
| Experiment 02b (corrected variants) | 6,448 | about $12 |
| Experiment 03, fitting the lens | one-off, 47 min on 4 A100s | about $8 |
| Experiment 03, reading 555 items | 527 GPU-seconds on 6 A100s | under $2 |
| Experiment 03, defended prompts (1,612) | 1,316 GPU-seconds | about $3 |
| Experiment 03, told-turn conversations (32) | under a minute of GPU | pennies |

Downloading the model costs no GPU time. Every GPU job resumes if interrupted: rerun the same command.

## Pipeline

```bash
# 0. data and model
npm install
npm run download                      # Microsoft's dataset into data/
pip install -r analysis/requirements.txt
pip install modal && modal setup
cd modal && modal run download_model.py   # 28 GB of Phi-3 into a Modal volume, one-off

# 1. experiment 01: reproduce the challenge
python3 analysis/ms_repeat_consistency.py     # how repeatable Microsoft's own runs were
python analysis/build_phase1_sample.py        # 400 attacks and their exact prompts
cd modal && modal run phase1_reproduce.py                                        # 8 answers each
modal run phase1_reproduce.py --condition undefended --temperature 0.7 --tag t07  # sensitivity
cd .. && python analysis/report_01_reproduction.py

# 2. experiment 02: the library as published, and Microsoft's spotlighting
python analysis/build_library_experiment.py   # prompt pieces for 200 attacks + 203 clean emails
node analysis/apply_library.mjs               # applies the published npm library
python3 analysis/make_run_list.py             # skips prompts identical to no-defense
cd modal && modal run generate.py \
  --sample runs/02-library-defenses/prompts_run.jsonl \
  --out runs/02-library-defenses/generations.jsonl --n-samples 4 --chunk 25

# 3. experiment 02b: corrected marker variants (needs the tokenizer option, 2.1.0-alpha+)
cd .. && node analysis/apply_library_v2.mjs
cd modal && modal run generate.py \
  --sample runs/02-library-defenses/prompts_v2.jsonl \
  --out runs/02-library-defenses/generations_v2.jsonl --n-samples 4 --chunk 25

# 4. reports (reads every generations file it finds)
cd .. && python analysis/report_02_library.py

# 5. experiment 03: the Jacobian lens on undefended prompts
python3 analysis/build_lens_sample.py         # 403 prompts + 152 recorded answers to replay
cd modal && modal run lens_fit.py             # fit the lens on WikiText, ~47 min, one-off
modal run lens_apply.py                       # read all 555 items, ~3 min
cd .. && python3 analysis/report_03_lens.py

# 6. experiment 03, defenses and the told turn
python3 analysis/build_lens_sample_defended.py && python3 analysis/build_lens_sample_told.py
cd modal && modal run lens_apply.py --sample runs/03-jacobian-lens/sample_defended.jsonl \
    --out runs/03-jacobian-lens/readouts_defended.jsonl --chunk 25
modal run lens_apply.py --sample runs/03-jacobian-lens/sample_told.jsonl \
    --out runs/03-jacobian-lens/readouts_told.jsonl --chunk 8
cd .. && python3 analysis/report_03_defenses.py && python3 analysis/report_03_told.py
```

### What the lens is

A **Jacobian lens** is one fixed linear map per layer that turns a hidden state into words, from
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens). `lens_fit.py` fits it on 100
plain-English WikiText passages — never on these prompts, so it is not tuned to the thing it
measures — and caches it at `/models/lens/phi3-wikitext100.pt` on the Modal volume. Fitting is the
expensive part and happens once; every readout after that reuses the same file.

`lens_fit.py` also runs depth probes on sentences whose answer is known, at three context lengths.
Layers 24, 32 and 38 pass; layers 8 and 16 return junk, which is why the report treats readings
below layer 20 as uninterpretable.

### Which copy of the library gets used

`analysis/apply_library.mjs` uses the **published** package in `node_modules`, which is the point:
experiment 02 measures the library as shipped. The recorded numbers came from `2.0.0-alpha`; if a newer
version is installed, that script warns, and every prompt records the version it was built with.

`analysis/apply_library_v2.mjs` needs the `tokenizer` option (2.1.0-alpha and later). It picks, in order:

1. `LIB_PATH=/path/to/index.js` if set
2. the installed `spotlighting-datamarking`, if it has the option
3. a sibling checkout at `../spotlighting-datamarking`

The winner's path, version and SHA-256 are written to `runs/02-library-defenses/library_provenance.json`
and copied into the report manifest. Check it without rebuilding anything:

```bash
node analysis/apply_library_v2.mjs --provenance-only
```

## Layout

| Path | What it holds |
|---|---|
| `analysis/` | Everything that runs on your machine: sample building, prompt building, reports |
| `modal/` | The GPU jobs. `generate.py` is the general runner; `msref/` holds Microsoft's files, unmodified |
| `modal/llmail_prompt.py` | A byte-faithful port of how the challenge agent prompted Phi-3 and scored tool calls |
| `modal/lens_fit.py` | Fits the Jacobian lens and runs its depth probes. One-off |
| `modal/lens_apply.py` | Reads the lens on prompts (study A) and along recorded answers (study B) |
| `results/` | Publishable output: reports, manifests, tables, figures |
| `runs/` | Raw prompts and model answers, git-ignored, pinned by checksum in the manifests |
| `src/` | The original Node harness. Not used by these experiments, see the caveat below |

## Caveats that shape the results

- **Phi-3 has no system role.** Its chat template drops system messages, so the challenge agent put
  everything in one user turn, and so does the port.
- **The original Node harness in `src/` differs from the challenge recipe** in six ways (query wording,
  missing benign email, no chat-token filtering, unsuffixed tool name, different sampling and token
  limits, system prompt sent separately). Prompts therefore come from `modal/llmail_prompt.py`.
- **Absolute attack rates here are lower than the challenge's**, roughly 0.7 times on undefended
  prompts. Experiment 01 rules out the token limit, the parser, temperature, model weights and prompt
  length; what remains is Azure's serving stack. Compare defenses with each other, not with the paper.
- **Experiment 03's main report reads undefended prompts.** The defended pass is its own report
  (`REPORT-defenses.md`) and reads two positions, not three: the benign email's end cannot be located
  under markers.
- **A lens reading is a probe, not the model's output.** It is what a linear map fitted on unrelated
  English extracts from one hidden state. A high reading on tool words does not mean the model has
  decided to call a tool.
- **Study B is partly circular past the tool call.** Teacher forcing replays the recorded answer, so
  once a winning answer has begun its tool call the reading is partly reading that call back. The
  report carries a second curve that keeps only positions before it.
