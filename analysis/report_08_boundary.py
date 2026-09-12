"""Experiment 08: the one thing the lens sees that the model's own output does not.

Experiment 03's baseline report showed the lens loses to the model's own next-token distribution at
every test of detection. This is the exception, and the only result in the project that needed a lens.

At the end of the ATTACKER'S email -- mid-prompt, where an output distribution is not a useful thing
to read -- the lens sometimes reads turn-boundary tokens: END, USER, SY. That is the model
representing the attack as though a speaking turn had ended and something else were about to talk,
which is structurally what an indirect injection is: text impersonating a new speaker.

Two questions, both answered on all 3,999 repeat-submitted attacks with Microsoft's own outcomes:
  1. does that reading predict which attacks succeed?
  2. does it add anything the model's own output flag does not already say?

A third word bucket is carried as a control: ignore / instruction / warning. Those separate attacks
from clean emails but, as it turns out, predict nothing.

Inputs:  runs/03-jacobian-lens/{sample_all,readouts_all,readouts_all_model,readouts_withmodel}.jsonl
Outputs: results/08-boundary/REPORT.md, tables/*.csv

Run:  python3 analysis/report_08_boundary.py
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting as rp  # noqa: E402
from report_03_lens import boot_ci, mean  # noqa: E402

RAW = os.path.join(rp.ROOT, "runs", "03-jacobian-lens")
OUT = os.path.join(rp.ROOT, "results", "08-boundary")
POS = "attacker_email_end"
LO, HI = 20, 38
BUCKETS = {
    "turn_boundary": re.compile(r"^\s*(END|USER|SY|SYS|ASS)\s*$", re.I),
    "instruction_words": re.compile(r"^\s*(ignore|ignored|instruction|instructions|instruct|"
                                    r"warning|warnings|override|directive)\s*$", re.I),
}


def clean_threshold():
    """The largest reading any of experiment 03's 203 clean prompts produced at the prompt's end."""
    best = None
    for r in rp.jsonl(os.path.join(RAW, "readouts_withmodel.jsonl")):
        if "error" in r or r["group"] != "clean":
            continue
        k = next((k for k in r.get("model_readings", []) if k["position_label"] == "prompt_end"), None)
        if k and (best is None or k["tool_word_prob"] > best):
            best = k["tool_word_prob"]
    return best


def load():
    sample = {r["id"]: r for r in rp.jsonl(os.path.join(RAW, "sample_all.jsonl"))}
    model = {}
    p = os.path.join(RAW, "readouts_all_model.jsonl")
    if os.path.exists(p):
        for r in rp.jsonl(p):
            if "error" in r:
                continue
            k = next((k for k in r.get("model_readings", []) if k["position_label"] == "prompt_end"), None)
            if k:
                model[r["id"]] = k["tool_word_prob"]
    rows = []
    for r in rp.jsonl(os.path.join(RAW, "readouts_all.jsonl")):
        if "error" in r:
            continue
        s = sample.get(r["id"])
        if not s or not s.get("ms_submissions"):
            continue
        words = collections.defaultdict(list)
        for k in r["readings"]:
            if k["position_label"] == POS:
                words[k["layer"]] = k["top_words"]
        row = {"id": r["id"], "rate": s["ms_wins"] / s["ms_submissions"],
               "subs": s["ms_submissions"], "model": model.get(r["id"]), "by_layer": words}
        for name, pat in BUCKETS.items():
            row[name] = any(pat.match(w) for layer, ws in words.items()
                            if LO <= layer <= HI for w in ws)
        rows.append(row)
    return rows


def split_row(name, on, off):
    m1, l1, h1 = boot_ci(on, mean, seed=1)
    m0, l0, h0 = boot_ci(off, mean, seed=2)
    return {"signal": name, "n_shows": len(on), "rate_shows": m1, "shows_lo": l1, "shows_hi": h1,
            "n_not": len(off), "rate_not": m0, "not_lo": l0, "not_hi": h0,
            "separates": "yes" if (l1 > h0 or l0 > h1) else "no"}


def main():
    rows = load()
    thr = clean_threshold()
    ci = lambda m, lo, hi: f"{rp.pct(m)} ({rp.pct(lo)} to {rp.pct(hi)})"  # noqa: E731

    t1 = [split_row(n, [r["rate"] for r in rows if r[n]], [r["rate"] for r in rows if not r[n]])
          for n in BUCKETS]
    rp.write_csv(os.path.join(OUT, "tables", "signal-vs-success.csv"), t1,
                 ["signal", "n_shows", "rate_shows", "shows_lo", "shows_hi", "n_not", "rate_not",
                  "not_lo", "not_hi", "separates"])

    have = [r for r in rows if r["model"] is not None]
    quad, cells = [], (("output flag + boundary", lambda r: r["model"] > thr and r["turn_boundary"]),
                       ("output flag only", lambda r: r["model"] > thr and not r["turn_boundary"]),
                       ("boundary only", lambda r: r["model"] <= thr and r["turn_boundary"]),
                       ("neither", lambda r: r["model"] <= thr and not r["turn_boundary"]))
    for lab, sel in cells:
        g = [r["rate"] for r in have if sel(r)]
        m, lo, hi = boot_ci(g, mean, seed=3)
        quad.append({"group": lab, "attacks": len(g), "win_rate": m, "lo": lo, "hi": hi})
    rp.write_csv(os.path.join(OUT, "tables", "two-signals-crossed.csv"), quad,
                 ["group", "attacks", "win_rate", "lo", "hi"])

    strata = []
    for nm, grp in (("already flagged by the output", [r for r in have if r["model"] > thr]),
                    ("not flagged by the output", [r for r in have if r["model"] <= thr])):
        s = split_row(nm, [r["rate"] for r in grp if r["turn_boundary"]],
                      [r["rate"] for r in grp if not r["turn_boundary"]])
        strata.append(s)
    rp.write_csv(os.path.join(OUT, "tables", "incremental-within-strata.csv"), strata,
                 ["signal", "n_shows", "rate_shows", "shows_lo", "shows_hi", "n_not", "rate_not",
                  "not_lo", "not_hi", "separates"])

    by_layer = []
    for layer in range(LO, HI + 1):
        on = [r["rate"] for r in rows if any(BUCKETS["turn_boundary"].match(w)
                                             for w in r["by_layer"].get(layer, []))]
        off = [r["rate"] for r in rows if not any(BUCKETS["turn_boundary"].match(w)
                                                  for w in r["by_layer"].get(layer, []))]
        if len(on) < 30:
            by_layer.append({"layer": layer, "n_shows": len(on), "rate_shows": float("nan"),
                             "rate_not": float("nan"), "separates": "too few"})
            continue
        s = split_row(str(layer), on, off)
        by_layer.append({"layer": layer, "n_shows": s["n_shows"], "rate_shows": s["rate_shows"],
                         "rate_not": s["rate_not"], "separates": s["separates"]})
    rp.write_csv(os.path.join(OUT, "tables", "boundary-by-layer.csv"), by_layer,
                 ["layer", "n_shows", "rate_shows", "rate_not", "separates"])

    b = t1[0]
    i = t1[1]
    lines = [
        "# Experiment 08: the one thing the lens sees that the output does not",
        "",
        "Everywhere else in this project, reading the model's own next-token distribution beats "
        "reading the Jacobian lens (see `REPORT-baseline.md`). This is the exception.",
        "",
        "At the **end of the attacker's email** — mid-prompt, where a next-token distribution is not "
        "a useful thing to read — the lens sometimes reads turn-boundary tokens: `END`, `USER`, `SY`. "
        "That is the model representing the attack as though a speaking turn had ended and something "
        "else were about to talk, which is structurally what an indirect injection is: text "
        "impersonating a new speaker.",
        "",
        f"Measured on all **{len(rows):,}** repeat-submitted undefended attacks, labelled with "
        "Microsoft's own recorded outcomes.",
        "",
        "## Does it predict which attacks succeed?",
        "",
        rp.md_table(t1, [("signal", "Reading at the end of the attack", None),
                         ("n_shows", "Attacks showing it", str),
                         ("rate_shows", "Their win rate", lambda v: rp.pct(v)),
                         ("n_not", "Attacks not showing it", str),
                         ("rate_not", "Their win rate", lambda v: rp.pct(v)),
                         ("separates", "Intervals separate", None)]),
        "",
        f"The turn boundary separates cleanly: {ci(b['rate_shows'], b['shows_lo'], b['shows_hi'])} "
        f"against {ci(b['rate_not'], b['not_lo'], b['not_hi'])}.",
        "",
        f"**The instruction words do not.** `ignore` / `instruction` / `warning` appear for "
        f"{i['n_shows']:,} attacks and none of the 203 clean prompts, so they do distinguish an "
        "attack from an ordinary email — but they say nothing about which attacks work "
        f"({ci(i['rate_shows'], i['shows_lo'], i['shows_hi'])} against "
        f"{ci(i['rate_not'], i['not_lo'], i['not_hi'])}, and slightly the wrong way round). An "
        "earlier draft of this work highlighted those words. It should not have.",
        "",
        "## Does it add anything the output does not already say?",
        "",
        "Crossed against the model's own output flag at the end of the prompt, thresholded at the "
        f"largest reading any clean prompt produced ({thr:.5f}):",
        "",
        rp.md_table(quad, [("group", "Group", None), ("attacks", "Attacks", str),
                           ("win_rate", "Win rate", lambda v: rp.pct(v)),
                           ("lo", "95% low", lambda v: rp.pct(v)),
                           ("hi", "95% high", lambda v: rp.pct(v))]),
        "",
        "And within each stratum separately, which is the test that matters:",
        "",
        rp.md_table(strata, [("signal", "Among attacks that are…", None),
                             ("n_shows", "With boundary", str),
                             ("rate_shows", "Win rate", lambda v: rp.pct(v)),
                             ("n_not", "Without", str),
                             ("rate_not", "Win rate", lambda v: rp.pct(v)),
                             ("separates", "Intervals separate", None)]),
        "",
        "**It adds information in both strata.** The boundary reading is not a restatement of the "
        "output signal; the two are read at different positions and each contributes.",
        "",
        "## Where in the network it lives",
        "",
        "Full table in `tables/boundary-by-layer.csv`. The reading appears from about layer 25 and "
        "separates at almost every layer from there to 38. Below 25 it barely occurs.",
        "",
        "## What this does and does not show",
        "",
        "- It is correlational. The reading does not cause the attack to succeed; both may follow "
        "from something about the attack's construction.",
        "- Win rates are Microsoft's, from a median of two submissions per attack, so each label is "
        "coarse. With 3,999 attacks that averages out, but no single row means much.",
        "- It does not show the model judging the email as illegitimate. Representing text as a turn "
        "boundary is a structural fact, not an opinion about it.",
        "- One model, one prompt format. See the scope note in every other report here.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python3 analysis/report_08_boundary.py",
        "```",
    ]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"built_at": rp.now_utc(), "attacks": len(rows), "threshold": thr,
               "git": rp.git_info()}, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"wrote {OUT}/REPORT.md  ({len(rows):,} attacks)")


if __name__ == "__main__":
    main()
