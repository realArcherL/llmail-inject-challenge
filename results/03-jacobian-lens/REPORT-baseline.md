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
- **Depth trades recall for precision.** The threshold columns show it plainly: the model's output flags 119 attacks that win 27.8%; layer 38 flags 77 that win 35.6%; layer 30 flags only 27 that win 55.1%. Every one of those splits has non-overlapping intervals against its own below-threshold group. The middle layers are a low-recall, high-precision flag: their AUROC is poor because most attacks sit below most clean prompts there, but the few they do surface are the ones most likely to succeed. AUROC is the wrong summary for them; the tail is the story.
- **What the lens is actually for here** is the depth profile — the signal is absent at layer 30, a third of the way there at 34, most of the way at 36 — and reading positions the model's own decoder cannot be pointed at, such as the end of the attacker's email at layers 22 to 28, where the injection-shape words appear. Those results do not have a baseline equivalent, because there is no output distribution at a middle layer.

## Reproducing

```bash
cd modal && modal run lens_apply.py --study A \
    --out runs/03-jacobian-lens/readouts_withmodel.jsonl --chunk 25
cd .. && python3 analysis/report_03_baseline.py
```
