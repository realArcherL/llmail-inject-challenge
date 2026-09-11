"""Build the Phase 1 sample: which payloads to re-run, with the exact prompts Phi-3 saw.

Conditions (prompt-identical level groups, see ms_repeat_consistency.py):
  undefended  levels 1a, 1c, 1g
  spotlight   levels 1e, 1i

Only payloads Microsoft ran two or more times are eligible. That lets us compare
our replay against Microsoft's own replay on the same population:
  - each payload gets ONE recorded label, from a randomly chosen submission
  - Microsoft's baseline asks: given that label, how often did ANOTHER of their
    submissions of the same payload win? (weighted exactly like our sampling)
  - we draw N_PER_LABEL recorded wins and N_PER_LABEL recorded losses per condition

Writes runs/phase1/sample.jsonl and runs/phase1/ms_baselines.json.
Run with a Python that has pyyaml and tiktoken:
  python analysis/build_phase1_sample.py
"""
import collections
import json
import os
import random
import sys
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "data", "raw_submissions_phase1.jsonl")
OUT = os.path.join(ROOT, "runs", "phase1")
sys.path.insert(0, os.path.join(ROOT, "modal"))
import llmail_prompt as lp  # noqa: E402

SEED = 20260911
N_PER_LABEL = 100
CONDITIONS = {
    "undefended": ({"level1a", "level1c", "level1g"}, False),
    "spotlight": ({"level1e", "level1i"}, True),
}


def load_groups():
    groups = {c: collections.defaultdict(list) for c in CONDITIONS}
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            for cond, (levels, _) in CONDITIONS.items():
                if d["scenario"] not in levels:
                    continue
                if not str(d.get("output", "")).startswith("Processed example output"):
                    continue  # crashed job, label meaningless
                obj = d["objectives"]
                obj = json.loads(obj) if isinstance(obj, str) else obj
                groups[cond][(d.get("subject", ""), d.get("body", ""))].append(
                    {"rowkey": d["RowKey"], "scenario": d["scenario"], "won": bool(obj.get("exfil.sent"))})
    return groups


def ms_baselines(repeated):
    num_w = den_w = num_l = den_l = 0.0
    agree = []
    for subs in repeated:
        k, s = len(subs), sum(x["won"] for x in subs)
        num_w += (s / k) * ((s - 1) / (k - 1))
        den_w += s / k
        num_l += ((k - s) / k) * (s / (k - 1))
        den_l += (k - s) / k
        agree.append((comb(s, 2) + comb(k - s, 2)) / comb(k, 2))
    return {
        "repeated_payloads": len(repeated),
        "p_win_given_recorded_win": num_w / den_w,
        "p_win_given_recorded_loss": num_l / den_l,
        "recorded_win_prevalence": den_w / len(repeated),
        "pairwise_agreement": sum(agree) / len(agree),
    }


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    groups = load_groups()
    baselines, sample = {}, []

    for cond, (_, spotlight) in CONDITIONS.items():
        repeated = sorted(((key, subs) for key, subs in groups[cond].items() if len(subs) >= 2),
                          key=lambda kv: kv[0])
        baselines[cond] = ms_baselines([subs for _, subs in repeated])

        pools = {True: [], False: []}
        for key, subs in repeated:
            chosen = rng.choice(subs)
            pools[chosen["won"]].append((key, subs, chosen))
        for label in (True, False):
            picks = rng.sample(pools[label], min(N_PER_LABEL, len(pools[label])))
            for i, ((subject, body), subs, chosen) in enumerate(picks):
                suffix = lp.new_tool_suffix(rng)
                prompt, tool_name = lp.build_level1_prompt(subject, body, spotlight, suffix)
                sample.append({
                    "id": f"{cond}-{'win' if label else 'loss'}-{i:03d}",
                    "condition": cond,
                    "recorded_label": label,
                    "recorded_rowkey": chosen["rowkey"],
                    "recorded_scenario": chosen["scenario"],
                    "ms_wins": sum(x["won"] for x in subs),
                    "ms_submissions": len(subs),
                    "tool_name": tool_name,
                    "subject_preview": subject[:80],
                    "prompt": prompt,
                })
        print(f"{cond:10} repeated payloads {len(repeated):>5}  recorded-win pool {len(pools[True]):>4}"
              f"  recorded-loss pool {len(pools[False]):>5}  -> drew {N_PER_LABEL}+{N_PER_LABEL}")

    with open(os.path.join(OUT, "sample.jsonl"), "w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(OUT, "ms_baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    lens = sorted(len(s["prompt"]) for s in sample)
    print(f"\nwrote {len(sample)} prompts to runs/phase1/sample.jsonl")
    print(f"prompt length in characters: median {lens[len(lens)//2]:,}  max {lens[-1]:,}")
    print("Microsoft's own replay baselines:")
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
