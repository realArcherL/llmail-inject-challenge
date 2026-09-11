"""Experiment 02 inputs: Phase 1's 200 undefended attacks plus Microsoft's 203 clean emails.

Writes runs/02-library-defenses/parts.jsonl, one row per base item, holding the raw prompt
pieces (system rules, tool prompt, query, formatted emails) and Microsoft's two reference
prompts: no defense, and Microsoft spotlighting. analysis/apply_library.mjs then applies the
spotlighting-datamarking library to the very same pieces.

Attacks reuse Phase 1's tool-name suffix per payload, so their no-defense prompt is
byte-identical to Phase 1 and Phase 1's 8 answers per attack serve as the baseline.

Run with a Python that has pyyaml and tiktoken:
  python analysis/build_library_experiment.py
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "data", "raw_submissions_phase1.jsonl")
PHASE1 = os.path.join(ROOT, "runs", "phase1", "sample.jsonl")
OUT_DIR = os.path.join(ROOT, "runs", "02-library-defenses")
sys.path.insert(0, os.path.join(ROOT, "modal"))
import llmail_prompt as lp  # noqa: E402

SEED = 20260912


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    phase1 = [json.loads(l) for l in open(PHASE1)]
    attacks = [s for s in phase1 if s["condition"] == "undefended"]
    want = {s["recorded_rowkey"] for s in attacks}
    rows = {}
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            if d["RowKey"] in want:
                rows[d["RowKey"]] = d

    out = []
    for s in attacks:
        d = rows[s["recorded_rowkey"]]
        parts = lp.level1_parts(lp.format_email(d["subject"], d["body"]), s["tool_name"].rsplit("_", 1)[1])
        prompt_none = lp.assemble(parts)
        assert prompt_none == s["prompt"], f"no-defense prompt drifted from Phase 1 for {s['id']}"
        out.append({
            "base_id": s["id"], "kind": "attack",
            "recorded_label": s["recorded_label"], "recorded_rowkey": s["recorded_rowkey"],
            "recorded_scenario": s["recorded_scenario"],
            "ms_wins": s["ms_wins"], "ms_submissions": s["ms_submissions"],
            "tool_name": parts["tool_name"], "parts": parts,
            "prompt_none": prompt_none, "prompt_ms_spotlight": lp.spotlight_microsoft(parts),
        })

    rng = random.Random(SEED)
    clean = json.load(open(os.path.join(ROOT, "modal", "msref", "fp_tests.json")))["emails"]
    for i, email in enumerate(clean):
        parts = lp.level1_parts(email, lp.new_tool_suffix(rng))
        out.append({
            "base_id": f"clean-{i:03d}", "kind": "clean", "clean_email": email,
            "tool_name": parts["tool_name"], "parts": parts,
            "prompt_none": lp.assemble(parts), "prompt_ms_spotlight": lp.spotlight_microsoft(parts),
        })

    path = os.path.join(OUT_DIR, "parts.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_att = sum(r["kind"] == "attack" for r in out)
    print(f"wrote {len(out)} base items ({n_att} attacks, {len(out) - n_att} clean emails) to runs/02-library-defenses/parts.jsonl")
    print("all attack no-defense prompts are byte-identical to Phase 1")


if __name__ == "__main__":
    main()
