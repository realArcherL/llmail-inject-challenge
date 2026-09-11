"""Measure the structural limits of the LLMail-Inject dataset.

Answers, from the raw submissions:
  1. Were attacks written for Phi-3 specifically, or reused across models?
  2. How often was the same payload retried?
  3. How much unique attack material is actually there?

Run:  python3 analysis/dataset_limits.py
"""
import json, hashlib, collections, os, sys

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "data")
PHI, GPT = set("acegi") | set("kmoqsu"), set("bdfhjt") | set("lnprv")


def model_of(level):
    suffix = level[-1]
    if suffix in PHI:
        return "phi3"
    if suffix in GPT:
        return "gpt"
    return "?"


def load(phase):
    path = os.path.join(DATA, f"raw_submissions_{phase}.jsonl")
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            obj = d["objectives"]
            if isinstance(obj, str):
                obj = json.loads(obj)
            yield d, obj


def main():
    # payload hash -> {model -> [success bools]}
    payload = collections.defaultdict(lambda: collections.defaultdict(list))
    team_models = collections.defaultdict(set)
    rows = 0

    for phase in ("phase1", "phase2"):
        for d, obj in load(phase):
            rows += 1
            h = hashlib.blake2b(
                (d.get("subject", "") + "\x00" + d.get("body", "")).encode(),
                digest_size=16,
            ).hexdigest()
            m = model_of(d["scenario"])
            payload[h][m].append(bool(obj.get("exfil.sent")))
            team_models[d.get("team_id")].add(m)

    uniq = len(payload)
    both = {h: v for h, v in payload.items() if "phi3" in v and "gpt" in v}
    only_phi = sum(1 for v in payload.values() if set(v) == {"phi3"})
    only_gpt = sum(1 for v in payload.values() if set(v) == {"gpt"})

    print(f"rows                       {rows:>10,}")
    print(f"unique payloads            {uniq:>10,}")
    print(f"  tried on Phi-3 only      {only_phi:>10,}  ({100*only_phi/uniq:.1f}%)")
    print(f"  tried on GPT only        {only_gpt:>10,}  ({100*only_gpt/uniq:.1f}%)")
    print(f"  tried on BOTH            {len(both):>10,}  ({100*len(both)/uniq:.1f}%)")

    # transfer: among payloads tried on both, does success carry over?
    tt = tf = ft = ff = 0
    for v in both.values():
        p, g = any(v["phi3"]), any(v["gpt"])
        if p and g: tt += 1
        elif p and not g: tf += 1
        elif not p and g: ft += 1
        else: ff += 1
    n = max(len(both), 1)
    print()
    print("among payloads tried on BOTH models (exfil.sent ever true):")
    print(f"  worked on both           {tt:>10,}  ({100*tt/n:.1f}%)")
    print(f"  Phi-3 only               {tf:>10,}  ({100*tf/n:.1f}%)")
    print(f"  GPT only                 {ft:>10,}  ({100*ft/n:.1f}%)")
    print(f"  worked on neither        {ff:>10,}  ({100*ff/n:.1f}%)")
    worked_phi = tt + tf
    if worked_phi:
        print(f"  transfer rate Phi-3 -> GPT {100*tt/worked_phi:>8.1f}%")

    # retries
    counts = collections.Counter(sum(len(x) for x in v.values()) for v in payload.values())
    once = counts[1]
    print()
    print(f"payload submitted exactly once   {once:>10,}  ({100*once/uniq:.1f}%)")
    print(f"payload submitted 2+ times       {uniq-once:>10,}  ({100*(uniq-once)/uniq:.1f}%)")
    print(f"max submissions of one payload   {max(counts):>10,}")

    single = sum(1 for v in team_models.values() if len(v) == 1)
    print()
    print(f"teams                            {len(team_models):>10,}")
    print(f"  targeted one model family only {single:>10,}  ({100*single/len(team_models):.1f}%)")


if __name__ == "__main__":
    main()
