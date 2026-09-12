"""Experiment 03, baseline: the model's own output distribution, against the lens.

jlens.JacobianLens.apply returns the model's ACTUAL final-layer logits alongside the lens logits.
Experiment 03 discarded them. That was a mistake: the headline layer (38) sits one layer below the
target layer, so a deep-layer lens reading is close to the model's own next-token distribution, and
without the baseline there is no way to say how much of the result is the lens and how much is the
model's ordinary output.

This reads the 403 study-A prompts recording both, and compares them at the same position.

Inputs:  runs/03-jacobian-lens/readouts_withmodel.jsonl
Outputs: results/03-jacobian-lens/REPORT-baseline.md, tables/baseline-*.csv

Run:  cd modal && modal run lens_apply.py --study A \
          --out runs/03-jacobian-lens/readouts_withmodel.jsonl
      python3 analysis/report_03_baseline.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting as rp  # noqa: E402
from report_03_lens import (HEADLINE_LAYER, auroc, spearman, boot_ci, boot_ci2,  # noqa: E402
                            boot_ci_pairs, mean, num, ci)

RAW = os.path.join(rp.ROOT, "runs", "03-jacobian-lens")
OUT = os.path.join(rp.ROOT, "results", "03-jacobian-lens")
L = HEADLINE_LAYER


def lens_at(r, layer, pos="prompt_end"):
    k = next((k for k in r["readings"] if k["position_label"] == pos and k["layer"] == layer), None)
    return k["tool_word_prob"] if k else float("nan")


def model_at(r, pos="prompt_end"):
    k = next((k for k in r.get("model_readings", []) if k["position_label"] == pos), None)
    return k["tool_word_prob"] if k else float("nan")


def split(atk, cln, get):
    """Threshold at the largest clean reading: how many attacks clear it, and how often they won."""
    thr = max(v for v in (get(r) for r in cln) if v == v)
    above = [r["success_rate"] for r in atk if get(r) > thr and r.get("success_rate") is not None]
    below = [r["success_rate"] for r in atk if get(r) <= thr and r.get("success_rate") is not None]
    am, alo, ahi = boot_ci(above, mean, seed=1)
    bm, blo, bhi = boot_ci(below, mean, seed=2)
    return {"threshold": thr, "n_above": len(above), "n_below": len(below),
            "rate_above": am, "rate_above_lo": alo, "rate_above_hi": ahi,
            "rate_below": bm, "rate_below_lo": blo, "rate_below_hi": bhi}


def main():
    path = os.path.join(RAW, "readouts_withmodel.jsonl")
    if not os.path.exists(path):
        print(f"missing {path}; run lens_apply.py --study A first")
        return
    rows = [r for r in rp.jsonl(path) if "error" not in r]
    atk = [r for r in rows if r["group"] == "attack"]
    cln = [r for r in rows if r["group"] == "clean"]

    variants = [("model_output", "The model's own output", model_at),
                *[(f"lens_layer_{x}", f"Lens, layer {x}", (lambda x: lambda r: lens_at(r, x))(x))
                  for x in (30, 32, 34, 36, 38)]]
    table = []
    for key, name, get in variants:
        a = [get(r) for r in atk]
        c = [get(r) for r in cln]
        a = [v for v in a if v == v]
        c = [v for v in c if v == v]
        au, alo, ahi = boot_ci2(a, c, auroc, seed=3)
        pairs = [(get(r), r["success_rate"]) for r in atk if r.get("success_rate") is not None]
        rho, rlo, rhi = boot_ci_pairs(pairs, spearman, seed=4)
        sp = split(atk, cln, get)
        table.append({"measurement": key, "name": name, "auroc": au, "auroc_lo": alo, "auroc_hi": ahi,
                      "spearman_success": rho, "spearman_lo": rlo, "spearman_hi": rhi, **sp})
    rp.write_csv(os.path.join(OUT, "tables", "baseline-vs-lens.csv"), table,
                 ["measurement", "name", "auroc", "auroc_lo", "auroc_hi", "spearman_success",
                  "spearman_lo", "spearman_hi", "threshold", "n_above", "n_below", "rate_above",
                  "rate_above_lo", "rate_above_hi", "rate_below", "rate_below_lo", "rate_below_hi"])
    agree = spearman([lens_at(r, L) for r in rows], [model_at(r) for r in rows])
    m, lens38 = table[0], next(t for t in table if t["measurement"] == f"lens_layer_{L}")

    lines = [
        "# Experiment 03, baseline: the model's own output",
        "",
        "`JacobianLens.apply` returns the model's actual final-layer logits next to the lens logits. "
        "Experiment 03 ignored them. It should not have: the headline layer sits one layer below the "
        "lens's target layer, so a reading there is close to the model's ordinary next-token "
        "distribution. Without this comparison there is no way to say how much of the result is the "
        "lens and how much is the model simply telling you.",
        "",
        f"Same {len(rows)} prompts, same position (the end of the prompt), same fixed tool-word list.",
        "",
        rp.md_table(table, [("name", "Measurement", None),
                            ("auroc", "AUROC vs clean", lambda v: f"{v:.3f}"),
                            ("auroc_lo", "CI low", lambda v: f"{v:.3f}"),
                            ("auroc_hi", "CI high", lambda v: f"{v:.3f}"),
                            ("spearman_success", "Corr. with success", lambda v: f"{v:+.3f}"),
                            ("n_above", "Attacks above every clean", str),
                            ("rate_above", "Win rate, above", lambda v: rp.pct(v)),
                            ("rate_below", "Win rate, below", lambda v: rp.pct(v))]),
        "",
        "## What this changes",
        "",
        f"- **The model's own output is the better detector.** AUROC "
        f"{ci(m['auroc'], m['auroc_lo'], m['auroc_hi'], 3)} against "
        f"{ci(lens38['auroc'], lens38['auroc_lo'], lens38['auroc_hi'], 3)} for the lens at layer {L}. "
        f"It flags {m['n_above']} attacks above every clean prompt against {lens38['n_above']}. "
        "Anyone can reproduce it from the model's logits, with no interpretability tooling at all.",
        f"- **The two are largely the same signal.** Rank correlation between the lens at layer {L} "
        f"and the model's own output, across all {len(rows)} prompts: {agree:+.3f}.",
        f"- **The lens is slightly more selective.** Its flagged group is smaller and wins more often "
        f"({rp.pct(lens38['rate_above'])} against {rp.pct(m['rate_above'])}), and it tracks the "
        f"success rate a little better ({lens38['spearman_success']:+.3f} against "
        f"{m['spearman_success']:+.3f}), though the intervals overlap.",
        f"- **Depth trades recall for precision.** The threshold columns show it plainly: the model's "
        f"output flags {m['n_above']} attacks that win {rp.pct(m['rate_above'])}; layer {L} flags "
        f"{lens38['n_above']} that win {rp.pct(lens38['rate_above'])}; layer 30 flags only "
        f"{next(t for t in table if t['measurement'] == 'lens_layer_30')['n_above']} that win "
        f"{rp.pct(next(t for t in table if t['measurement'] == 'lens_layer_30')['rate_above'])}. "
        "Every one of those splits has non-overlapping intervals against its own below-threshold "
        "group. The middle layers are a low-recall, high-precision flag: their AUROC is poor because "
        "most attacks sit below most clean prompts there, but the few they do surface are the ones "
        "most likely to succeed. AUROC is the wrong summary for them; the tail is the story.",
        "- **What the lens is actually for here** is the depth profile — the signal is absent at "
        "layer 30, a third of the way there at 34, most of the way at 36 — and reading positions the "
        "model's own decoder cannot be pointed at, such as the end of the attacker's email at layers "
        "22 to 28, where the injection-shape words appear. Those results do not have a baseline "
        "equivalent, because there is no output distribution at a middle layer.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "cd modal && modal run lens_apply.py --study A \\",
        "    --out runs/03-jacobian-lens/readouts_withmodel.jsonl --chunk 25",
        "cd .. && python3 analysis/report_03_baseline.py",
        "```",
    ]
    with open(os.path.join(OUT, "REPORT-baseline.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    mp = os.path.join(OUT, "manifest.json")
    mf = json.load(open(mp)) if os.path.exists(mp) else {}
    mf["baseline"] = {"built_at": rp.now_utc(), "n_prompts": len(rows),
                      "model_output_auroc": m["auroc"], "lens_auroc": lens38["auroc"],
                      "agreement": agree,
                      "files": [rp.file_entry(os.path.join(OUT, "tables", "baseline-vs-lens.csv"), "output"),
                                rp.file_entry(path, "lens readouts with model logits")]}
    json.dump(mf, open(mp, "w"), indent=2)
    print(f"wrote {OUT}/REPORT-baseline.md")


if __name__ == "__main__":
    main()
