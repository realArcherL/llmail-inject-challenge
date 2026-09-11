# Modal box for Phi-3-medium

## Current pipeline

One A100 container runs everything with plain transformers. No vLLM.

| Step | Command | What it does |
|---|---|---|
| Weights | `modal run download_model.py` | one-off, 28 GB into volume `phi3-weights` |
| Phase 0 | `modal run spike_lens.py` | checks jlens loads Phi-3. Passed on 2026-09-11 |
| Phase 1 sample | `python analysis/build_phase1_sample.py` | picks 400 payloads, writes exact challenge prompts. Needs pyyaml and tiktoken |
| Phase 1 run | `modal run phase1_reproduce.py` | 8 samples per prompt, challenge settings. Smoke test: `--limit 4 --n-samples 2 --tag smoke` |
| Phase 1 report | `python3 analysis/phase1_report.py --tag full` | our replay against Microsoft's own replay |
| Exp. 02 inputs | `python analysis/build_library_experiment.py`, then `node analysis/apply_library.mjs`, then `python3 analysis/make_run_list.py` | prompt pieces; prompts defended by the real `spotlighting-datamarking` package; the list to generate |
| Exp. 02 run | `modal run generate.py --sample runs/02-library-defenses/prompts_run.jsonl --out runs/02-library-defenses/generations.jsonl --n-samples 4 --chunk 25` | Phi-3 answers every defended prompt; resumes if interrupted |
| Reports | `python analysis/report_01_reproduction.py` and `python analysis/report_02_library.py` | publishable `results/<experiment>/`: REPORT.md, manifest.json, tables, figures |

Supporting files:

- `hfload.py` loads Phi-3 on transformers 5.5+, which jlens requires. It patches one RoPE key in memory; files on the volume are untouched.
- `generate.py` is the generic generation job for every experiment after 01. `phase1_reproduce.py` stays unchanged as the exact code behind experiment 01.
- `llmail_prompt.py` is a faithful port of how the challenge agent prompted Phi-3 and scored tool calls. Level 1 only.
- `msref/` holds unmodified Microsoft files the port reads, MIT licensed, with the source commit in its README.

Phi-3 has no system role. Its chat template silently drops system messages, so the whole prompt goes in one user turn, exactly as Microsoft's agent did.

## Parked, not used

`serve_vllm.py`, `probe_endpoint.py` and `jacobian_probe.py` belong to the earlier two-GPU plan. They still use the old loading code and would hit the Phi-3 config error. The notes below describe that plan.

## Original notes for the parked files

Order of operations:

1. `pip3 install modal && modal setup`
2. Create Modal secret `huggingface-secret` with key `HF_TOKEN` (dashboard -> Secrets).
3. `modal run download_model.py`        # one-off, 28 GB into volume "phi3-weights"
4. `modal deploy serve_vllm.py`         # prints URL A  (OpenAI-compatible, L40S)
5. `modal deploy probe_endpoint.py`     # prints URL B  (gradient probe, A100)
6. `modal run jacobian_probe.py`        # sanity check without deploying anything

Node harness `.env`:

    LMSTUDIO_BASE_URL=<URL A>/v1
    LMSTUDIO_MODEL=phi3-medium
    LMSTUDIO_API_KEY=anything
    LMSTUDIO_TIMEOUT_MS=300000          # first call includes ~2 min cold start
    PROBE_URL=<URL B>

Calling the probe from JS:

    const r = await fetch(process.env.PROBE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, target: 'send' }),
    });
    const { target_prob, top5, per_token } = await r.json();

`messages` is the same array you already send to /chat/completions,
so the probe sees exactly the prompt the benchmark saw.
