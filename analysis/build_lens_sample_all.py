"""Experiment 03, robustness: the lens on EVERY repeat-submitted undefended attack, not a sample of 200.

Experiment 03 read 200 attacks. This reads all payloads that Microsoft ran two or more times on
undefended Phi-3 (the same population experiment 01 sampled from) at the end of the prompt only,
one forward pass each. No answers are generated: the label is Microsoft's own outcome, the share of
that payload's recorded submissions that succeeded.

The question is whether the tail found on 200 attacks (attacks that read above every clean email,
and fire more often) is a property of the population or of the sample.

Writes runs/03-jacobian-lens/sample_all.jsonl
Run:  uv run --with pyyaml --with tiktoken python3 analysis/build_lens_sample_all.py
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "modal"))
import build_phase1_sample as p1  # noqa: E402
import build_lens_sample as ls  # noqa: E402
import llmail_prompt as lp  # noqa: E402

OUT = os.path.join(ROOT, "runs", "03-jacobian-lens", "sample_all.jsonl")
SEED = 3  # tool-name suffixes only; the population is every repeated payload, nothing is drawn


def main():
    rng = random.Random(SEED)
    groups = p1.load_groups()
    repeated = sorted(((key, subs) for key, subs in groups["undefended"].items() if len(subs) >= 2),
                      key=lambda kv: kv[0])
    out = []
    for i, ((subject, body), subs) in enumerate(repeated):
        suffix = lp.new_tool_suffix(rng)
        prompt, tool_name = lp.build_level1_prompt(subject, body, False, suffix)
        wins = sum(x["won"] for x in subs)
        out.append({
            "id": f"A-all-{i:04d}", "study": "A", "group": "attack", "base_id": f"all-{i:04d}",
            "success_rate": wins / len(subs), "runs": len(subs), "ms_wins": wins,
            "ms_submissions": len(subs), "tool_name": tool_name,
            "spans": ls.spans(prompt), "prompt": prompt,
        })
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    ever = sum(1 for r in out if r["ms_wins"] > 0)
    print(f"wrote {len(out)} undefended attacks to {os.path.relpath(OUT, ROOT)}: "
          f"{ever} ever won for Microsoft, {len(out) - ever} never; "
          f"median submissions per payload {sorted(r['runs'] for r in out)[len(out)//2]}")
    lens = sorted(len(r["prompt"]) for r in out)
    print(f"prompt length in characters: median {lens[len(lens)//2]:,}  max {lens[-1]:,}")


if __name__ == "__main__":
    main()
