"""Experiment 03b inputs: the same 403 prompts as study A, but with a defense applied.

Study A read undefended prompts at the position that produces the first generated token and found
that attacks separate from clean emails there (AUROC 0.81). This sample asks the obvious next
question: does a defense change what the model has in mind at that position?

Four conditions from experiment 02, chosen to span the story:
  ms_spotlight         Microsoft's own spotlighting (query/email tags + fixed datamark)
  lib_markdata         the spotlighting-datamarking library as published (random marker between words)
  lib_base64           the library's base64 mode
  v2_uni_random_phi3   the fixed library: short private-use marker at Phi-3's own token boundaries

Each attack carries two success rates: undefended (8 runs, experiment 01) and under that defense
(4 runs, experiment 02). Only the end of the attacker's email and the end of the prompt are read;
the benign email's end cannot be located reliably once markers are in.

Writes runs/03-jacobian-lens/sample_defended.jsonl
Run:  python3 analysis/build_lens_sample_defended.py
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW02 = os.path.join(ROOT, "runs", "02-library-defenses")
PHASE1 = os.path.join(ROOT, "runs", "phase1", "results_full.jsonl")
OUT = os.path.join(ROOT, "runs", "03-jacobian-lens", "sample_defended.jsonl")

CONDITIONS = {
    "ms_spotlight": "prompts_run.jsonl",
    "lib_markdata": "prompts_run.jsonl",
    "lib_base64": "prompts_run.jsonl",
    "v2_uni_random_phi3": "prompts_v2.jsonl",
}


def jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def attacker_end(prompt, condition):
    """Last character of the email block. Microsoft's spotlighting closes it with a tag; the
    library conditions end the prompt with the (marked) attacker email itself."""
    if condition == "ms_spotlight":
        at = prompt.rfind("</emails_block")
        return at if at > 0 else len(prompt)
    return len(prompt)


def main():
    undefended = collections.defaultdict(list)
    for r in jsonl(PHASE1):
        if r["id"].startswith("undefended"):
            undefended[r["id"]].append(bool(r["exfil.sent"]))
    defended = collections.defaultdict(list)
    for fn in ("generations.jsonl", "generations_v2.jsonl"):
        p = os.path.join(RAW02, fn)
        if os.path.exists(p):
            for r in jsonl(p):
                defended[(r["base_id"], r["condition"])].append(bool(r["exfil.sent"]))

    prompts = {}
    for fn in set(CONDITIONS.values()):
        for r in jsonl(os.path.join(RAW02, fn)):
            if r["condition"] in CONDITIONS:
                prompts[(r["base_id"], r["condition"])] = r

    out = []
    for (base_id, cond), r in sorted(prompts.items()):
        u = undefended.get(base_id)
        d = defended.get((base_id, cond))
        out.append({
            "id": f"A-{base_id}__{cond}", "study": "A", "group": r["kind"], "base_id": base_id,
            "condition": cond, "data_marker": r.get("data_marker"),
            "success_rate": (sum(d) / len(d)) if d else None,
            "runs": len(d) if d else 0,
            "undefended_success_rate": (sum(u) / len(u)) if u else None,
            "spans": {"attacker_end": attacker_end(r["prompt"], cond)},
            "prompt": r["prompt"],
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} items to {os.path.relpath(OUT, ROOT)}")
    for cond in CONDITIONS:
        rows = [x for x in out if x["condition"] == cond]
        atk = [x for x in rows if x["group"] == "attack"]
        with_rate = [x for x in atk if x["success_rate"] is not None]
        mean = (sum(x["success_rate"] for x in with_rate) / len(with_rate)) if with_rate else float("nan")
        print(f"  {cond:20} {len(rows)} prompts, {len(atk)} attacks, "
              f"{len(with_rate)} with a defended success rate (mean {mean:.3f})")


if __name__ == "__main__":
    main()
