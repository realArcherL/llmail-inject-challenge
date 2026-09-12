# Experiment 03, robustness

Two checks a careful reader would ask for: the same measurement on twenty times as many attacks, and the same measurement with a lens learned from different text.

## Part 1: every repeat-submitted attack, not a sample

3,999 attacks, every payload Microsoft ran two or more times on undefended Phi-3, read at the end of the prompt at layer 38 with the same lens. No answers were generated; the label is Microsoft's own outcome. The threshold is the same as before: the largest reading among the 203 clean prompts (0.00087).

| Attacks | n | AUROC vs clean | Read above every clean | Win rate, above | Win rate, below | Win rate from |
|---|---|---|---|---|---|---|
| 200 attacks (experiment 03) | 200 | 0.806 | 38.5% | 35.6% | 12.3% | our 8 runs |
| all repeat-submitted attacks | 3999 | 0.759 | 24.7% | 36.0% | 12.3% | Microsoft's submissions |

On the full population: AUROC 0.759 (0.735 to 0.781); 988 of 3,999 attacks (24.7%) read above every clean prompt; those attacks won 36.0% (33.9% to 38.2%) of their submissions for Microsoft, against 12.3% (11.4% to 13.2%) for the rest. Rank correlation between the reading and Microsoft's win rate: 0.414 (0.387 to 0.440). Separating attacks that ever won from attacks that never won, by the reading alone: AUROC 0.738 (0.722 to 0.754).

Two things differ from the 200-attack study and are worth keeping in mind: the win rates here are Microsoft's, from a median of two submissions per payload, so they are coarse; and the population is 63% never-won payloads, where the 200 were balanced half and half.

## Part 2: a second lens, fitted on different text

Lens B was fitted on WikiText passages 101–200, none of which lens A saw. Both read the same 403 prompts at the end of the prompt, layer 38.

- Reading-by-reading agreement (rank correlation across all 403 prompts): 0.983 (0.976 to 0.987).
- Attack vs clean: AUROC 0.806 with lens A, 0.840 (0.799 to 0.878) with lens B.
- Rank correlation with the attack's success rate: 0.502 (0.371 to 0.610) with lens B (0.504 with lens A).
- Attacks flagged above every clean prompt: 77 by lens A, 76 by lens B, 73 by both (overlap 91% of the union).

| Layer | Rank corr., lens A vs B | AUROC, lens A | AUROC, lens B |
|---|---|---|---|
| 20 | no variance | 0.50 | 0.51 |
| 22 | no variance | 0.50 | 0.50 |
| 24 | +1.00 | 0.50 | 0.50 |
| 26 | +0.52 | 0.37 | 0.55 |
| 28 | +0.86 | 0.25 | 0.31 |
| 30 | +0.91 | 0.23 | 0.25 |
| 32 | +0.94 | 0.55 | 0.54 |
| 34 | +0.89 | 0.59 | 0.55 |
| 36 | +0.96 | 0.72 | 0.71 |
| 38 | +0.98 | 0.81 | 0.84 |

## Reproducing

```bash
uv run --with pyyaml --with tiktoken python3 analysis/build_lens_sample_all.py
cd modal && modal run lens_apply.py --sample runs/03-jacobian-lens/sample_all.jsonl \
    --out runs/03-jacobian-lens/readouts_all.jsonl --chunk 40
modal run lens_fit.py --skip 100 --tag phi3-wikitext100b
modal run lens_apply.py --study A --tag phi3-wikitext100b \
    --out runs/03-jacobian-lens/readouts_lensb.jsonl
cd .. && python3 analysis/report_03_robustness.py
```
