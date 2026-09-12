"""Build the run inspector: every prompt, every answer, every score, as a browsable page.

One file per condition so the page loads only what is being looked at. The prompt text is stored
once per prompt and shared by that prompt's runs.

Inputs:  runs/phase1/{sample,results_full}.jsonl
         runs/02-library-defenses/{prompts_run,prompts_v2,generations,generations_v2}.jsonl
Outputs: results/browser/{index.js,g-*.js}  (index.html is checked in alongside)

Run:  python3 analysis/build_transcript_browser.py
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "browser")
ORDER = ["phase1_undefended", "none", "phase1_spotlight", "ms_spotlight", "lib_markdata",
         "lib_randommark", "lib_base64", "v2_uni_spaces", "v2_uni_random_phi3",
         "v2_alnum_random_phi3", "v2_uni_words", "lib_sanitize"]
NAMES = {
    "phase1_undefended": "No defense (experiment 01, 8 runs)",
    "phase1_spotlight": "Microsoft spotlighting (experiment 01, 8 runs)",
    "none": "No defense", "ms_spotlight": "Microsoft spotlighting",
    "lib_sanitize": "Library: sanitize only", "lib_markdata": "Library: marker between words",
    "lib_randommark": "Library: marker at random points", "lib_base64": "Library: base64",
    "v2_uni_spaces": "Fixed: short Unicode in spaces",
    "v2_uni_random_phi3": "Fixed: short Unicode, Phi-3 points",
    "v2_alnum_random_phi3": "Fixed: alphanumeric, Phi-3 points",
    "v2_uni_words": "Fixed: short Unicode, word gaps only",
}


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def run_row(r, prompt_row, kind, label):
    return {"k": r.get("base_id", r.get("id")), "s": r["sample"], "resp": r["response"],
            "sent": bool(r["exfil.sent"]), "call": bool(r["tool_call_text_present"]),
            "dest": bool(r["exfil.destination"]), "body": bool(r["exfil.content"]),
            "pt": r["prompt_tokens"], "nt": r["new_tokens"], "tool": prompt_row["tool_name"],
            "kind": kind, "label": label, "calls": json.dumps(r["tool_calls"])[:600]}


def main():
    os.makedirs(OUT, exist_ok=True)
    groups = collections.defaultdict(lambda: {"prompts": {}, "runs": []})

    p1 = {r["id"]: r for r in jl(os.path.join(ROOT, "runs", "phase1", "sample.jsonl"))}
    for r in jl(os.path.join(ROOT, "runs", "phase1", "results_full.jsonl")):
        s = p1.get(r["id"])
        if not s:
            continue
        g = "phase1_" + ("undefended" if s["condition"] == "undefended" else "spotlight")
        groups[g]["prompts"][r["id"]] = s["prompt"]
        groups[g]["runs"].append(run_row(r, s, "attack", s["recorded_label"]))

    raw2 = os.path.join(ROOT, "runs", "02-library-defenses")
    for pf, gf in (("prompts_run.jsonl", "generations.jsonl"),
                   ("prompts_v2.jsonl", "generations_v2.jsonl")):
        if not os.path.exists(os.path.join(raw2, gf)):
            continue
        pr = {(x["base_id"], x["condition"]): x for x in jl(os.path.join(raw2, pf))}
        for r in jl(os.path.join(raw2, gf)):
            s = pr.get((r["base_id"], r["condition"]))
            if not s:
                continue
            groups[r["condition"]]["prompts"][r["base_id"]] = s["prompt"]
            groups[r["condition"]]["runs"].append(
                run_row(r, s, r["kind"], r.get("recorded_label")))

    index, total = [], 0
    for g, d in groups.items():
        fn = f"g-{g}.js"
        body = json.dumps({"prompts": d["prompts"], "runs": d["runs"]}, separators=(",", ":"))
        with open(os.path.join(OUT, fn), "w") as f:
            f.write(f"window.G=window.G||{{}};window.G[{json.dumps(g)}]={body};")
        mb = os.path.getsize(os.path.join(OUT, fn)) / 1e6
        total += mb
        attacks = sum(1 for r in d["runs"] if r["kind"] == "attack")
        sent = sum(1 for r in d["runs"] if r["sent"])
        index.append({"id": g, "name": NAMES.get(g, g), "file": fn, "runs": len(d["runs"]),
                      "prompts": len(d["prompts"]), "attacks": attacks,
                      "clean": len(d["runs"]) - attacks, "sent": sent,
                      "rate": (sent / attacks) if attacks else 0.0, "mb": round(mb, 1)})
    index.sort(key=lambda g: ORDER.index(g["id"]) if g["id"] in ORDER else 99)
    with open(os.path.join(OUT, "index.js"), "w") as f:
        f.write("window.IDX=" + json.dumps(index) + ";")
    print(f"wrote {len(index)} condition files to results/browser, {total:.1f} MB total, "
          f"{sum(g['runs'] for g in index):,} runs")
    for g in index:
        print(f"  {g['id']:24} {g['runs']:>5} runs  {g['sent']:>4} injected  {g['mb']:>5.1f} MB")


if __name__ == "__main__":
    main()
