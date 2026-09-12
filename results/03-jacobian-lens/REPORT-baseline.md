# Experiment 03, baseline: the model's own output

`JacobianLens.apply` returns the model's actual final-layer logits next to the lens logits. Experiment 03 ignored them. It should not have: the headline layer sits one layer below the lens's target layer, so a reading there is close to the model's ordinary next-token distribution. Without this comparison there is no way to say how much of the result is the lens and how much is the model simply telling you.

Same 403 prompts, same position (the end of the prompt), same fixed tool-word list.

| Measurement | AUROC vs clean | CI low | CI high | Corr. with success | Attacks above every clean | Win rate, above | Win rate, below |
|---|---|---|---|---|---|---|---|
| The model's own output | 0.848 | 0.807 | 0.890 | +0.449 | 119 | 27.8% | 11.6% |
| Lens, layer 30 | 0.232 | 0.180 | 0.286 | +0.019 | 27 | 55.1% | 16.0% |
| Lens, layer 32 | 0.546 | 0.486 | 0.603 | +0.077 | 30 | 47.9% | 16.5% |
| Lens, layer 34 | 0.588 | 0.531 | 0.647 | +0.294 | 46 | 43.8% | 14.5% |
| Lens, layer 36 | 0.720 | 0.670 | 0.772 | +0.337 | 44 | 44.0% | 14.8% |
| Lens, layer 38 | 0.806 | 0.759 | 0.849 | +0.504 | 77 | 35.6% | 12.3% |

## What this changes

- **The model's own output is the better detector.** AUROC 0.848 (0.807 to 0.890) against 0.806 (0.759 to 0.849) for the lens at layer 38. It flags 119 attacks above every clean prompt against 77. Anyone can reproduce it from the model's logits, with no interpretability tooling at all.
- **The two are largely the same signal.** Rank correlation between the lens at layer 38 and the model's own output, across all 403 prompts: +0.793.
- **The lens is slightly more selective.** Its flagged group is smaller and wins more often (35.6% against 27.8%), and it tracks the success rate a little better (+0.504 against +0.449), though the intervals overlap.
- **The apparent precision of the deep layers is the threshold, not the depth.** Flag the same NUMBER of attacks with each signal and ask what share of them fire: at 27 flags the model's output gets 55.1% and the best lens layer 55.6%; at 77 flags, 36.0% against 35.6%; at 119, 27.8% against 29.4%. Matched flag-for-flag the output equals or beats every layer. An earlier version of this report claimed depth bought precision. It does not, and the claim has been removed.
- **The one thing the lens shows that the output does not** is at the end of the attacker's email. Neither signal separates attacks from clean prompts by tool-word probability there (AUROC 0.50 for both), but the WORDS differ: the instruction and turn-boundary signature appears in the model's own top-8 for 27 of 200 attacks (4 of 203 clean), and in the lens's for 76 of 200 (2 of 203). Fifty attacks show it to the lens and not to the output. That is a description of what the model represents mid-prompt, not a detector, and it is the whole of the lens's unique contribution in this work.

## Reproducing

```bash
cd modal && modal run lens_apply.py --study A \
    --out runs/03-jacobian-lens/readouts_withmodel.jsonl --chunk 25
cd .. && python3 analysis/report_03_baseline.py
```
