# Results

Publishable outputs for the LLMail-Inject × Phi-3 study. Each experiment folder holds:

| File | What it is |
|---|---|
| `REPORT.md` | The write-up: question, setup, results, caveats, how to reproduce. Every number is computed by a script. |
| `manifest.json` | Provenance: git commit, model fingerprint, generation settings, and SHA-256 of every raw input and output. |
| `tables/*.csv` | Every number in the report, plus per-attack detail, for your own charts. |
| `figures/*.svg`, `*.png` | Figures in a light and a `_dark` variant. |

Raw prompts and model answers live in `runs/` (git-ignored, large). The manifests pin them by checksum.

| Experiment | Question |
|---|---|
| [01-reproduction](01-reproduction/REPORT.md) | Does our Phi-3 setup reproduce the challenge's recorded outcomes? |
| [02-library-defenses](02-library-defenses/REPORT.md) | Does the spotlighting-datamarking library stop the attacks, versus Microsoft's spotlighting, and at what cost? |

Regenerate any folder with `python analysis/report_<experiment>.py`.
