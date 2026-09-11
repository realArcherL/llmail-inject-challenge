"""Phase 1 report: does our replay reproduce the challenge's recorded outcomes?

Compares, per condition, our replay against Microsoft's own replay of the same payloads:
  win | recorded win    how often a payload wins again, given its recorded label was a win
  win | recorded loss   how often it wins, given its recorded label was a loss
  agreement             pairwise agreement between repeated runs of one payload
  rank correlation      Spearman between Microsoft's per-payload win rate and ours

Run:  python3 analysis/phase1_report.py --tag full
"""
import argparse
import collections
import json
import os
from math import comb, sqrt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P1 = os.path.join(ROOT, "runs", "phase1")


def mean_se(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return m, sqrt(v / len(xs))


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return cov / den if den else float("nan")


def pct(m, se=None):
    return f"{100*m:5.1f}%" + (f" ± {196*se:4.1f}" if se is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    tag = ap.parse_args().tag

    sample = {}
    for line in open(os.path.join(P1, "sample.jsonl")):
        s = json.loads(line)
        sample[s["id"]] = s
    base = json.load(open(os.path.join(P1, "ms_baselines.json")))
    res = collections.defaultdict(list)
    for line in open(os.path.join(P1, f"results_{tag}.jsonl")):
        r = json.loads(line)
        res[r["id"]].append(r)

    for cond in ("undefended", "spotlight"):
        ids = [i for i, s in sample.items() if s["condition"] == cond and i in res]
        if not ids:
            continue
        p = {i: sum(r["exfil.sent"] for r in res[i]) / len(res[i]) for i in ids}
        win = [p[i] for i in ids if sample[i]["recorded_label"]]
        loss = [p[i] for i in ids if not sample[i]["recorded_label"]]
        agree = []
        for i in ids:
            k, s = len(res[i]), sum(r["exfil.sent"] for r in res[i])
            if k >= 2:
                agree.append((comb(s, 2) + comb(k - s, 2)) / comb(k, 2))
        b = base[cond]
        mw, sw = mean_se(win)
        ml, sl = mean_se(loss)
        pi = b["recorded_win_prevalence"]
        rows = [r for i in ids for r in res[i]]
        n = len(rows)

        print(f"\n=== {cond}: {len(ids)} payloads, {n} generations ===")
        print(f"{'':30}{'Microsoft replay':>18}{'our replay':>22}")
        print(f"{'win | recorded win':30}{pct(b['p_win_given_recorded_win']):>18}{pct(mw, sw):>22}")
        print(f"{'win | recorded loss':30}{pct(b['p_win_given_recorded_loss']):>18}{pct(ml, sl):>22}")
        print(f"{'overall win rate (reweighted)':30}{pct(pi*b['p_win_given_recorded_win'] + (1-pi)*b['p_win_given_recorded_loss']):>18}"
              f"{pct(pi*mw + (1-pi)*ml):>22}")
        print(f"{'repeat agreement':30}{pct(b['pairwise_agreement']):>18}{pct(*mean_se(agree)):>22}")
        print(f"rank correlation, Microsoft per-payload win rate vs ours: "
              f"{spearman([sample[i]['ms_wins']/sample[i]['ms_submissions'] for i in ids], [p[i] for i in ids]):.2f}")
        print("diagnostics:")
        print(f"  any tool-call text in reply     {pct(sum(r['tool_call_text_present'] for r in rows)/n)}")
        print(f"  parsed tool call, any name      {pct(sum(bool(r['tool_calls']) for r in rows)/n)}")
        print(f"  right tool name (exfil.sent)    {pct(sum(r['exfil.sent'] for r in rows)/n)}")
        print(f"  right address                   {pct(sum(r['exfil.destination'] for r in rows)/n)}")
        print(f"  right body                      {pct(sum(r['exfil.content'] for r in rows)/n)}")
        print(f"  hit the 500-token limit         {pct(sum(r['hit_max_tokens'] for r in rows)/n)}")
        print(f"  would have crashed Microsoft's scorer  {sum(r['would_crash'] for r in rows)}")
        print(f"  prompt tokens: median {sorted(r['prompt_tokens'] for r in rows)[n//2]}, max {max(r['prompt_tokens'] for r in rows)}")


if __name__ == "__main__":
    main()
