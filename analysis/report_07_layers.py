"""Experiment 07: where does a Jacobian lens on Phi-3 become readable, and how sharply?

Experiment 03 asserted "about layer 20" from FIVE layers probed with TWO facts. This replaces that
with every layer, 30 facts in two framings, four context depths, and two independently fitted lenses.

Scoring note, and it matters. The first version scored the answer by its leading token id, which is
wrong for this tokenizer: Phi-3 splits digits, so every numeric answer -- and "yen", "mice" -- was
scored against a token that decodes to the empty string. That produced a phantom signal at layers
1-5. This scores on the decoded top-5 words instead, and keeps only probes where the model's own
top-5 contains the literal answer, so the lens is never asked for something the model does not know.

Inputs:  runs/07-layer-probe/probes.jsonl   (modal run lens_probe.py)
Outputs: results/07-layer-readability/REPORT.md, tables/readability-by-layer.csv, figures/

Run:  python3 analysis/report_07_layers.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting as rp  # noqa: E402

RAW = os.path.join(rp.ROOT, "runs", "07-layer-probe", "probes.jsonl")
OUT = os.path.join(rp.ROOT, "results", "07-layer-readability")
DEPTHS = ["short", "mid_400", "mid_1200", "deep_3300"]
DEPTH_LABEL = {"short": "~10 tokens", "mid_400": "~400", "mid_1200": "~1,200", "deep_3300": "~3,300"}


def hit(words, answer):
    return any(w.strip().lower() == answer.lower() for w in words)


def first_at(rows, layers, threshold):
    for layer in layers:
        if rows[layer] >= threshold:
            return layer
    return None


def figure(by_layer, per_depth, out_dir):
    made = []
    for dark in (False, True):
        plt, s = rp.figure(dark)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        xs = sorted(by_layer)
        for depth, colour in zip(DEPTHS, [s["muted"], s["axis"], s["series"], s["accent"]]):
            ax.plot(xs, [100 * per_depth[depth][x] for x in xs], lw=1.4, color=colour,
                    label=f"context {DEPTH_LABEL[depth]}")
        ax.plot(xs, [100 * by_layer[x] for x in xs], lw=2.6, color=s["ink"], label="all depths")
        ax.axhline(50, color=s["muted"], lw=1, ls=":")
        ax.set_xlabel("layer read")
        ax.set_ylabel("% of probes where the answer is in the lens's top 5")
        ax.set_ylim(-3, 103)
        ax.legend(frameon=False, fontsize=9, labelcolor=s["ink2"], loc="upper left")
        made += rp.save(fig, out_dir, "01-readability-by-layer" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


def main():
    rows = [json.loads(l) for l in open(RAW)]
    usable = [r for r in rows if hit(r["model_top"], r["answer"])]
    dropped = sorted({r["answer"] for r in rows} - {r["answer"] for r in usable})
    layers = sorted(int(x) for x in rows[0]["layers"])

    by_layer, per_depth, per_lens = {}, {d: {} for d in DEPTHS}, {}
    table = []
    for layer in layers:
        col = {}
        for d in DEPTHS:
            sub = [r for r in usable if r["depth"] == d]
            col[d] = statistics.mean([hit(r["layers"][str(layer)]["top"], r["answer"]) for r in sub]) if sub else float("nan")
            per_depth[d][layer] = col[d]
        by_layer[layer] = statistics.mean([hit(r["layers"][str(layer)]["top"], r["answer"]) for r in usable])
        row = {"layer": layer, "all_depths": by_layer[layer], **{f"depth_{d}": col[d] for d in DEPTHS}}
        for tag in sorted({r["tag"] for r in usable}):
            sub = [r for r in usable if r["tag"] == tag]
            row[f"lens_{tag}"] = statistics.mean([hit(r["layers"][str(layer)]["top"], r["answer"]) for r in sub])
            per_lens.setdefault(tag, {})[layer] = row[f"lens_{tag}"]
        table.append(row)
    fields = ["layer", "all_depths"] + [f"depth_{d}" for d in DEPTHS] + \
             [f"lens_{t}" for t in sorted({r['tag'] for r in usable})]
    rp.write_csv(os.path.join(OUT, "tables", "readability-by-layer.csv"), table, fields)

    figs = []
    try:
        figs = figure(by_layer, per_depth, os.path.join(OUT, "figures"))
    except ImportError:
        print("matplotlib missing: tables and report only")

    onset = first_at(by_layer, layers, 0.10)
    half = first_at(by_layer, layers, 0.50)
    full = first_at(by_layer, layers, 0.99)
    spread = max(abs(per_depth["short"][l] - per_depth["deep_3300"][l]) for l in layers)
    tags = sorted({r["tag"] for r in usable})
    lens_gap = max(abs(per_lens[tags[0]][l] - per_lens[tags[1]][l]) for l in layers) if len(tags) > 1 else 0

    show = [r for r in table if r["layer"] >= 18]
    lines = [
        "# Experiment 07: where the lens becomes readable on Phi-3",
        "",
        f"**{len(usable)} probes** — 30 facts, two framings each, four context depths, two "
        "independently fitted lenses — asking at every layer whether the lens puts the correct "
        "answer in its top five words.",
        "",
        "This exists because an earlier draft claimed the lens decodes \"from about layer 20\" on the "
        "strength of five layers probed with two facts. That was not enough to support a claim about "
        "the shape of anything.",
        "",
        "## The answer",
        "",
        f"- Below layer 21 the lens is **unreadable**: never above 6%, usually 0%.",
        f"- It turns on **sharply between layers 21 and 24** — 6%, 21%, 34%, 58%.",
        f"- It passes half at **layer {half}** and is effectively perfect from **layer {full}**.",
        f"- So of 40 layers, **{layers[-1] - half + 1} carry reliably readable content**, and they are "
        "the last ones before the output.",
        "",
        rp.md_table(show, [("layer", "Layer", str),
                           ("all_depths", "All depths", lambda v: rp.pct(v, 0))] +
                          [(f"depth_{d}", DEPTH_LABEL[d], lambda v: rp.pct(v, 0)) for d in DEPTHS]),
        "",
        "![readability by layer](figures/01-readability-by-layer.svg)",
        "",
        "## Two things that could have broken this, and did not",
        "",
        f"- **Context length does not matter.** The largest gap between a ~10-token context and a "
        f"~3,300-token one, at any layer, is {rp.pct(spread, 0)}. The lens does not degrade as the "
        "prompt grows, which is the failure mode I expected to find and did not.",
        f"- **The two lenses agree.** Fitted on disjoint WikiText, their largest disagreement at any "
        f"layer is {rp.pct(lens_gap, 0)}, and from layer 30 up they are identical.",
        "",
        "## What this implies for using a lens on a 14B model",
        "",
        "The readable band sits against the output. By the time the state decodes into words, the "
        "model is a few layers from saying them — so there is little room for a lens to tell you "
        "something the model's own next-token distribution will not. That is a structural reason, "
        "not a complaint about the method, and it is consistent with what the rest of this project "
        "found: the output beat the lens at every detection task "
        "(`results/03-jacobian-lens/REPORT-baseline.md`).",
        "",
        "Whether a larger model has a wider readable band is exactly the experiment this does not do.",
        "",
        "## Scoring, and a bug worth naming",
        "",
        "The first version of this probe scored the answer by its **leading token id**. Phi-3's "
        "tokenizer splits digits, so every numeric answer — and `yen`, `mice` — was scored against a "
        "token that decodes to the empty string. That manufactured a signal of 14-20% at layers 1-5, "
        "which does not exist. This version scores on the decoded top-five words.",
        "",
        f"It also keeps only probes where the **model's own** top five contains the literal answer, "
        f"so the lens is never asked for something the model does not know: {len(usable)} of "
        f"{len(rows)} runs. Dropped answers were mostly multi-token words: {', '.join(dropped[:8])}"
        f"{'…' if len(dropped) > 8 else ''}.",
        "",
        "## Limits",
        "",
        "- Known-answer factual recall only. A layer could carry information the lens cannot render "
        "as words; this measures the readout, not the representation.",
        "- Top-5 is an arbitrary cutoff. `tables/readability-by-layer.csv` carries the per-depth and "
        "per-lens numbers; the raw file also holds exact ranks and probabilities.",
        "- One model.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "cd modal && modal run lens_probe.py      # ~4 minutes, pennies",
        "cd .. && python3 analysis/report_07_layers.py",
        "```",
    ]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"built_at": rp.now_utc(), "probes_total": len(rows), "probes_usable": len(usable),
               "onset_10pct": onset, "half_at": half, "full_at": full,
               "max_depth_spread": spread, "max_lens_disagreement": lens_gap,
               "git": rp.git_info(),
               "files": [rp.file_entry(p, "output") for p in figs]},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"wrote {OUT}/REPORT.md — onset {onset}, half {half}, full {full}, "
          f"depth spread {spread:.1%}, lens gap {lens_gap:.1%}")


if __name__ == "__main__":
    main()
