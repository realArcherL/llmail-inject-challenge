"""Experiment 03 inputs: which prompts and which recorded answers the Jacobian lens reads.

Study A, prompt level. Every undefended attack prompt (200) and every clean-email prompt (203), read
at the end of the prompt and at the end of each email. The model's state while reading a prompt is
fixed, so this cannot explain why one run fired and another did not; it can only track an attack's
propensity. So each attack carries its measured success rate over 8 recorded runs.

Study B, run level. 76 attacks that both won and lost on the identical prompt. For each, one winning
answer and one losing answer, read along the answer itself. Same prompt, same wording, different
outcome: the only thing that varies is what the model generated.

Writes runs/03-jacobian-lens/sample.jsonl
Run:  python3 analysis/build_lens_sample.py
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1 = os.path.join(ROOT, "runs", "phase1", "results_full.jsonl")
PARTS = os.path.join(ROOT, "runs", "02-library-defenses", "parts.jsonl")
OUT_DIR = os.path.join(ROOT, "runs", "03-jacobian-lens")
MAX_PROMPT_TOKENS = 1500  # keeps the read close to the lens's fitting length


def spans(prompt):
    """Character offsets of the two emails inside the level 1 prompt."""
    first = prompt.find(" email: ")
    last = prompt.rfind(" email: ")
    return {"benign_start": first, "attacker_start": last, "attacker_end": len(prompt)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = {json.loads(l)["base_id"]: json.loads(l) for l in open(PARTS)}
    runs = collections.defaultdict(list)
    for line in open(PHASE1):
        r = json.loads(line)
        if r["id"].startswith("undefended"):
            runs[r["id"]].append(r)

    out = []
    for base_id, p in parts.items():
        prompt = p["prompt_none"]
        rs = sorted(runs.get(base_id, []), key=lambda r: r["sample"])
        rate = (sum(bool(r["exfil.sent"]) for r in rs) / len(rs)) if rs else None
        out.append({
            "id": f"A-{base_id}", "study": "A", "group": p["kind"], "base_id": base_id,
            "success_rate": rate, "runs": len(rs),
            "prompt_tokens": rs[0]["prompt_tokens"] if rs else None,
            "spans": spans(prompt), "prompt": prompt,
        })

    pairs = 0
    for base_id, rs in sorted(runs.items()):
        if not rs or rs[0]["prompt_tokens"] > MAX_PROMPT_TOKENS:
            continue
        rs = sorted(rs, key=lambda r: r["sample"])
        win = next((r for r in rs if r["exfil.sent"]), None)
        # a clean miss: no tool call at all, not merely a malformed one
        miss = next((r for r in rs if not r["exfil.sent"] and not r["tool_call_text_present"]), None)
        if not (win and miss):
            continue
        pairs += 1
        rate = sum(bool(r["exfil.sent"]) for r in rs) / len(rs)
        for outcome, r in (("win", win), ("miss", miss)):
            tool_at = r["response"].find('{"type": "function"')
            out.append({
                "id": f"B-{base_id}-{outcome}", "study": "B", "group": "attack", "base_id": base_id,
                "outcome": outcome, "sample": r["sample"], "success_rate": rate,
                "prompt_tokens": r["prompt_tokens"], "response_tokens": r["new_tokens"],
                "tool_call_char": tool_at if tool_at >= 0 else None,
                "spans": spans(parts[base_id]["prompt_none"]),
                "prompt": parts[base_id]["prompt_none"], "response": r["response"],
            })

    path = os.path.join(OUT_DIR, "sample.jsonl")
    with open(path, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    a = [r for r in out if r["study"] == "A"]
    print(f"wrote {len(out)} readouts to runs/03-jacobian-lens/sample.jsonl")
    print(f"  study A: {sum(1 for r in a if r['group'] == 'attack')} attack prompts, "
          f"{sum(1 for r in a if r['group'] == 'clean')} clean prompts")
    print(f"  study B: {pairs} attacks with a winning and a losing answer on the same prompt "
          f"({2 * pairs} readouts)")
    rates = [r["success_rate"] for r in a if r["group"] == "attack" and r["success_rate"] is not None]
    print(f"  attack success rates span {min(rates):.2f} to {max(rates):.2f}, "
          f"{sum(1 for x in rates if 0 < x < 1)} strictly in between")


if __name__ == "__main__":
    main()
