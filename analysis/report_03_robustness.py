"""Experiment 03, robustness: does the headline survive a 20x larger population and a second lens?

Part 1, population. Experiment 03 read 200 attacks. runs/03-jacobian-lens/readouts_all.jsonl reads
every payload Microsoft ran two or more times on undefended Phi-3 (3,999), same lens, same position
and layer, no answers generated: the label is Microsoft's own win rate over that payload's
submissions. The same 203 clean prompts set the threshold.

Part 2, lens. A second lens fitted on the NEXT 100 WikiText passages (disjoint from the first) reads
the same 403 prompts. If the readings and the flagged set agree, the result is not an artefact of
which 100 passages the lens happened to learn from.

Inputs:  readouts.jsonl (lens A, 200 attacks + 203 clean), readouts_all.jsonl (lens A, 3,999 attacks),
         readouts_lensb.jsonl (lens B, 403 prompts), sample_all.jsonl
Outputs: results/03-jacobian-lens/REPORT-robustness.md, tables/robustness-*.csv

Run:  python3 analysis/report_03_robustness.py
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


def reading(r, label="prompt_end", layer=L):
    k = next((k for k in r["readings"] if k["position_label"] == label and k["layer"] == layer), None)
    return k["tool_word_prob"] if k else float("nan")


def load(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        return [], []
    rows = rp.jsonl(p)
    return [r for r in rows if "error" not in r], [r for r in rows if "error" in r]


def part_population(base, lines, tables):
    rows, errors = load("readouts_all.jsonl")
    if not rows:
        lines += ["## Part 1: every repeat-submitted attack", "", "_Not run yet._", ""]
        return
    sample = {r["id"]: r for r in rp.jsonl(os.path.join(RAW, "sample_all.jsonl"))}
    expected = len(sample)
    clean = [reading(r) for r in base if r["group"] == "clean"]
    clean = [v for v in clean if v == v]
    thr = max(clean)
    atk = [(reading(r), r.get("success_rate"), sample.get(r["id"], {}).get("ms_wins", 0) > 0) for r in rows]
    atk = [a for a in atk if a[0] == a[0]]
    vals = [a[0] for a in atk]
    au, alo, ahi = boot_ci2(vals, clean, auroc, seed=11)
    above = [a for a in atk if a[0] > thr]
    below = [a for a in atk if a[0] <= thr]
    am, al, ah = rp.bootstrap_mean([a[1] for a in above])
    bm, bl, bh = rp.bootstrap_mean([a[1] for a in below])
    rho, rlo, rhi = boot_ci_pairs([(a[0], a[1]) for a in atk], spearman, seed=12)
    ever = [a[0] for a in atk if a[2]]
    never = [a[0] for a in atk if not a[2]]
    au2, a2lo, a2hi = boot_ci2(ever, never, auroc, seed=13)
    # the 200-attack numbers, for the side-by-side
    b_atk = [(reading(r), r.get("success_rate")) for r in base if r["group"] == "attack"]
    b_vals = [v for v, _ in b_atk]
    b_au = auroc(b_vals, clean)
    b_above = [s for v, s in b_atk if v > thr]
    b_below = [s for v, s in b_atk if v <= thr]
    tables.append(("robustness-population.csv", [
        {"set": "200 attacks (experiment 03)", "n": len(b_atk), "auroc_vs_clean": b_au,
         "share_above_every_clean": len(b_above) / len(b_atk), "win_rate_above": mean(b_above),
         "win_rate_below": mean(b_below), "label": "our 8 runs"},
        {"set": "all repeat-submitted attacks", "n": len(atk), "auroc_vs_clean": au,
         "share_above_every_clean": len(above) / len(atk), "win_rate_above": am,
         "win_rate_below": bm, "label": "Microsoft's submissions"},
    ], ["set", "n", "auroc_vs_clean", "share_above_every_clean", "win_rate_above", "win_rate_below", "label"]))
    complete = len(rows) == expected and not errors
    lines += [
        "## Part 1: every repeat-submitted attack, not a sample",
        "",
        ("" if complete else f"> **Status: incomplete.** {len(rows)} of {expected} read, {len(errors)} failed.\n\n") +
        f"{len(atk):,} attacks, every payload Microsoft ran two or more times on undefended Phi-3, read at "
        f"the end of the prompt at layer {L} with the same lens. No answers were generated; the label is "
        "Microsoft's own outcome. The threshold is the same as before: the largest reading among the 203 "
        f"clean prompts ({num(thr)}).",
        "",
        rp.md_table(tables[-1][1], [("set", "Attacks", None), ("n", "n", str),
                                    ("auroc_vs_clean", "AUROC vs clean", lambda v: f"{v:.3f}"),
                                    ("share_above_every_clean", "Read above every clean", lambda v: rp.pct(v)),
                                    ("win_rate_above", "Win rate, above", lambda v: rp.pct(v)),
                                    ("win_rate_below", "Win rate, below", lambda v: rp.pct(v)),
                                    ("label", "Win rate from", None)]),
        "",
        f"On the full population: AUROC {ci(au, alo, ahi, 3)}; {len(above):,} of {len(atk):,} attacks "
        f"({rp.pct(len(above) / len(atk))}) read above every clean prompt; those attacks won "
        f"{rp.pct(am)} ({rp.pct(al)} to {rp.pct(ah)}) of their submissions for Microsoft, against "
        f"{rp.pct(bm)} ({rp.pct(bl)} to {rp.pct(bh)}) for the rest. Rank correlation between the reading and "
        f"Microsoft's win rate: {ci(rho, rlo, rhi, 3)}. Separating attacks that ever won from attacks that "
        f"never won, by the reading alone: AUROC {ci(au2, a2lo, a2hi, 3)}.",
        "",
        "Two things differ from the 200-attack study and are worth keeping in mind: the win rates here "
        "are Microsoft's, from a median of two submissions per payload, so they are coarse; and the "
        "population is 63% never-won payloads, where the 200 were balanced half and half.",
        "",
    ]


def part_lens(base, lines, tables):
    rows, errors = load("readouts_lensb.jsonl")
    if not rows:
        lines += ["## Part 2: a second lens, fitted on different text", "", "_Not run yet._", ""]
        return
    A = {r["id"]: r for r in base}
    B = {r["id"]: r for r in rows}
    ids = sorted(set(A) & set(B))
    pairs = [(reading(A[i]), reading(B[i])) for i in ids]
    pairs = [p for p in pairs if p[0] == p[0] and p[1] == p[1]]
    rho, rlo, rhi = boot_ci_pairs(pairs, spearman, seed=21)
    atk_b = [reading(B[i]) for i in ids if B[i]["group"] == "attack"]
    cln_b = [reading(B[i]) for i in ids if B[i]["group"] == "clean"]
    au_b, blo, bhi = boot_ci2(atk_b, cln_b, auroc, seed=22)
    atk_a = [reading(A[i]) for i in ids if A[i]["group"] == "attack"]
    cln_a = [reading(A[i]) for i in ids if A[i]["group"] == "clean"]
    au_a = auroc(atk_a, cln_a)
    thr_a, thr_b = max(cln_a), max(cln_b)
    flag_a = {i for i in ids if A[i]["group"] == "attack" and reading(A[i]) > thr_a}
    flag_b = {i for i in ids if B[i]["group"] == "attack" and reading(B[i]) > thr_b}
    jacc = len(flag_a & flag_b) / max(1, len(flag_a | flag_b))
    rates_b = [(reading(B[i]), B[i].get("success_rate")) for i in ids
               if B[i]["group"] == "attack" and B[i].get("success_rate") is not None]
    rho_b, rblo, rbhi = boot_ci_pairs(rates_b, spearman, seed=23)
    # agreement layer by layer, for the table
    by_layer = []
    for layer in range(20, 39, 2):
        pr = [(reading(A[i], layer=layer), reading(B[i], layer=layer)) for i in ids]
        pr = [p for p in pr if p[0] == p[0] and p[1] == p[1]]
        aa = auroc([reading(A[i], layer=layer) for i in ids if A[i]["group"] == "attack"],
                   [reading(A[i], layer=layer) for i in ids if A[i]["group"] == "clean"])
        ab = auroc([reading(B[i], layer=layer) for i in ids if B[i]["group"] == "attack"],
                   [reading(B[i], layer=layer) for i in ids if B[i]["group"] == "clean"])
        by_layer.append({"layer": layer, "rank_corr_lensA_lensB": spearman([p[0] for p in pr], [p[1] for p in pr]),
                         "auroc_lensA": aa, "auroc_lensB": ab})
    tables.append(("robustness-two-lenses-by-layer.csv", by_layer,
                   ["layer", "rank_corr_lensA_lensB", "auroc_lensA", "auroc_lensB"]))
    lines += [
        "## Part 2: a second lens, fitted on different text",
        "",
        ("" if not errors else f"> {len(errors)} readouts failed under lens B.\n\n") +
        f"Lens B was fitted on WikiText passages 101–200, none of which lens A saw. Both read the same "
        f"{len(ids)} prompts at the end of the prompt, layer {L}.",
        "",
        f"- Reading-by-reading agreement (rank correlation across all {len(pairs)} prompts): {ci(rho, rlo, rhi, 3)}.",
        f"- Attack vs clean: AUROC {au_a:.3f} with lens A, {ci(au_b, blo, bhi, 3)} with lens B.",
        f"- Rank correlation with the attack's success rate: {ci(rho_b, rblo, rbhi, 3)} with lens B "
        "(0.504 with lens A).",
        f"- Attacks flagged above every clean prompt: {len(flag_a)} by lens A, {len(flag_b)} by lens B, "
        f"{len(flag_a & flag_b)} by both (overlap {rp.pct(jacc, 0)} of the union).",
        "",
        rp.md_table(by_layer, [("layer", "Layer", str),
                               ("rank_corr_lensA_lensB", "Rank corr., lens A vs B",
                                lambda v: "no variance" if v != v else f"{v:+.2f}"),
                               ("auroc_lensA", "AUROC, lens A", lambda v: f"{v:.2f}"),
                               ("auroc_lensB", "AUROC, lens B", lambda v: f"{v:.2f}")]),
        "",
    ]


def main():
    base, _ = load("readouts.jsonl")
    base = [r for r in base if r["study"] == "A"]
    lines = ["# Experiment 03, robustness", "",
             "Two checks a careful reader would ask for: the same measurement on twenty times as many "
             "attacks, and the same measurement with a lens learned from different text.", ""]
    tables = []
    part_population(base, lines, tables)
    part_lens(base, lines, tables)
    lines += ["## Reproducing", "", "```bash",
              "uv run --with pyyaml --with tiktoken python3 analysis/build_lens_sample_all.py",
              "cd modal && modal run lens_apply.py --sample runs/03-jacobian-lens/sample_all.jsonl \\",
              "    --out runs/03-jacobian-lens/readouts_all.jsonl --chunk 40",
              "modal run lens_fit.py --skip 100 --tag phi3-wikitext100b",
              "modal run lens_apply.py --study A --tag phi3-wikitext100b \\",
              "    --out runs/03-jacobian-lens/readouts_lensb.jsonl",
              "cd .. && python3 analysis/report_03_robustness.py", "```"]
    for name, rows, fields in tables:
        rp.write_csv(os.path.join(OUT, "tables", name), rows, fields)
    with open(os.path.join(OUT, "REPORT-robustness.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    mp = os.path.join(OUT, "manifest.json")
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    m["robustness"] = {"built_at": rp.now_utc(),
                       "files": [rp.file_entry(os.path.join(OUT, "tables", n), "output") for n, _, _ in tables] +
                                [rp.file_entry(os.path.join(RAW, n), "lens readouts") for n in
                                 ("readouts_all.jsonl", "readouts_lensb.jsonl") if os.path.exists(os.path.join(RAW, n))]}
    json.dump(m, open(mp, "w"), indent=2)
    print(f"wrote {OUT}/REPORT-robustness.md")


if __name__ == "__main__":
    main()
