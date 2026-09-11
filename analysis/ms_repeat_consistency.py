"""How repeatable was Phi-3 inside the original challenge?

Levels 1a (Prompt Shield), 1c (TaskTracker) and 1g (LLM Judge) send Phi-3 the
exact same prompt. Those defenses only raise a detection flag; they never change
the model input (microsoft/llmail-inject-challenge, src/agent/workloads/llm.py,
LLMWithDefenses.prompt). Levels 1e and 1i add spotlighting to the prompt.

So a payload submitted more than once to 1a/1c/1g is the same prompt sampled
again on Microsoft's own infrastructure. How often those repeats agree is the
ceiling for any reproduction: nobody can match their labels more often than
their own system matched itself.

Run:  python3 analysis/ms_repeat_consistency.py
"""
import collections, json, os
from math import comb

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "data", "raw_submissions_phase1.jsonl")
CONDITIONS = {
    "undefended prompt (1a, 1c, 1g)": {"level1a", "level1c", "level1g"},
    "spotlight prompt (1e, 1i)": {"level1e", "level1i"},
}


def main():
    per = {name: {"rows": 0, "failed": 0, "groups": collections.defaultdict(list)} for name in CONDITIONS}
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            for name, levels in CONDITIONS.items():
                if d["scenario"] not in levels:
                    continue
                st = per[name]
                st["rows"] += 1
                if not str(d.get("output", "")).startswith("Processed example output"):
                    st["failed"] += 1          # job crashed; its labels mean nothing
                    continue
                obj = d["objectives"]
                obj = json.loads(obj) if isinstance(obj, str) else obj
                st["groups"][(d.get("subject", ""), d.get("body", ""))].append(bool(obj.get("exfil.sent")))

    for name, st in per.items():
        groups = st["groups"]
        kept = sum(len(v) for v in groups.values())
        succ = sum(sum(v) for v in groups.values())
        p = succ / kept
        rep = [v for v in groups.values() if len(v) >= 2]
        per_payload = [(comb(sum(v), 2) + comb(len(v) - sum(v), 2)) / comb(len(v), 2) for v in rep]
        mixed = sum(1 for v in rep if 0 < sum(v) < len(v))
        any_win = [v for v in rep if sum(v) > 0]
        always = sum(1 for v in any_win if all(v))
        biggest = max(rep, key=len) if rep else []

        print(f"\n=== {name} ===")
        print(f"rows                               {st['rows']:>8,}")
        print(f"  dropped, job crashed             {st['failed']:>8,}")
        print(f"  usable                           {kept:>8,}   success rate {100*p:.1f}%")
        print(f"unique payloads                    {len(groups):>8,}")
        print(f"  submitted 2+ times               {len(rep):>8,}")
        if rep:
            print(f"repeat agreement, per payload avg  {100*sum(per_payload)/len(per_payload):>7.1f}%")
            print(f"  if outcomes were coin flips at   {100*(p*p+(1-p)*(1-p)):>7.1f}%   (same success rate, no memory)")
            print(f"  repeated payloads with mixed     {100*mixed/len(rep):>7.1f}%   (won sometimes, lost sometimes)")
            if any_win:
                print(f"  of payloads that won at least once, won every time  {100*always/len(any_win):.1f}%")
            print(f"  most-repeated payload submitted  {len(biggest):>8,} times")


if __name__ == "__main__":
    main()
