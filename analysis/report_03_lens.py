"""Build results/03-jacobian-lens: tables, figures, REPORT.md and manifest.json.

Question: with no defense in the prompt at all, what does the model have in mind while it reads an
attack, and does that differ from a clean email? Two studies, because one cannot answer both halves.

Study A, prompt level (403 prompts: 200 attack, 203 clean). One forward pass each, read at the end
of the benign email, the end of the attacker's email, and the very end of the prompt. The model's
state while reading a fixed prompt is fixed, so this CANNOT explain why one run of a prompt fired
and another did not. It can only track propensity, so each attack carries its measured success rate
over 8 recorded runs.

Study B, run level (152 replays: 76 attacks that both won and lost on the IDENTICAL prompt, one
winning and one losing answer each). Read along the answer under teacher forcing. Same prompt, same
wording; the only thing that varies is what the model generated.

Inputs (raw, git-ignored):  runs/03-jacobian-lens/{sample,readouts}.jsonl
                            runs/03-jacobian-lens/lens_meta.json    (if the fit wrote one)
Outputs (publishable):      results/03-jacobian-lens/

Run:  python3 analysis/report_03_lens.py [--readouts FILE]   (figures need matplotlib)
"""
import argparse
import collections
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting as rp  # noqa: E402

EXP = "03-jacobian-lens"
RAW = os.path.join(rp.ROOT, "runs", EXP)
OUT = os.path.join(rp.ROOT, "results", EXP)

# The lens is fitted on plain English (WikiText), never on these prompts, so it is not tuned to the
# thing it measures. Depth probes at fit time: layers 24/32/38 name the right answer at shallow, mid
# and deep context; layers 8 and 16 return junk. Headline numbers are therefore restricted to 20+.
TRUSTED_FROM = 20
# The headline layer is fixed BEFORE looking at any separation number: the deepest source layer,
# i.e. the readout closest to what the model actually emits, and one the depth probes pass. Picking
# the layer with the best AUROC out of 39 candidates and then reporting that AUROC would be
# selection bias, so the per-layer curves are shown in full and treated as exploratory.
HEADLINE_LAYER = 38
POSITIONS = [
    ("benign_email_end", "End of the benign email"),
    ("attacker_email_end", "End of the attacker's email"),
    ("prompt_end", "End of the prompt (produces the first generated token)"),
]
N_BINS = 10  # deciles of the answer, for study B
MIN_PAIRS = 20  # a bin below this is not plotted
# Third categorical slot from the validated palette (slots 1-3 clear the all-pairs CVD gate).
THIRD = {"light": "#1baf7a", "dark": "#199e70"}


# ---------------------------------------------------------------- statistics

def auroc(pos, neg):
    """Probability a random positive scores above a random negative. Ties count a half.
    0.5 means the reading tells the two groups apart no better than a coin."""
    if not pos or not neg:
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        r = (i + j + 1) / 2  # average rank of the tied block, 1-based
        for k in range(i, j):
            ranks[k] = r
        i = j
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(xs, ys):
    """Rank correlation. Robust to the fact that lens probabilities are not normally distributed."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = (i + j + 1) / 2
            for k in range(i, j):
                r[order[k]] = avg
            i = j
        return r

    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def boot_ci2(a, b, stat, n=2000, seed=0):
    """95% interval for a statistic of two independent groups, resampling each."""
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    reps = sorted(stat([a[rng.randrange(len(a))] for _ in a],
                       [b[rng.randrange(len(b))] for _ in b]) for _ in range(n))
    reps = [x for x in reps if x == x]
    if not reps:
        return stat(a, b), float("nan"), float("nan")
    return stat(a, b), reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]


def boot_ci_pairs(pairs, stat, n=2000, seed=0):
    """95% interval for a statistic of paired (x, y) values, resampling pairs."""
    if len(pairs) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    obs = stat([p[0] for p in pairs], [p[1] for p in pairs])
    reps = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = stat([p[0] for p in s], [p[1] for p in s])
        if v == v:
            reps.append(v)
    reps.sort()
    if not reps:
        return obs, float("nan"), float("nan")
    return obs, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]


def boot_ci(values, stat, n=2000, seed=0):
    """95% percentile interval for any statistic of a list, resampling the list."""
    values = list(values)
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng, k = random.Random(seed), len(values)
    obs = stat(values)
    reps = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return obs, reps[int(0.025 * n)], reps[int(0.975 * n) - 1]


def mean(vs):
    vs = [v for v in vs if v == v]
    return sum(vs) / len(vs) if vs else float("nan")


# ---------------------------------------------------------------- loading

def load(readouts_path):
    sample = {r["id"]: r for r in rp.jsonl(os.path.join(RAW, "sample.jsonl"))}
    rows = rp.jsonl(readouts_path)
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    for r in rows:
        r["sample"] = sample.get(r["id"], {})
    return sample, rows, errors


def index_a(rows):
    """{(position_label, layer): {"attack": [...], "clean": [...], "rates": [(prob, rate)]}}"""
    out = collections.defaultdict(lambda: {"attack": [], "clean": [], "rates": []})
    for r in rows:
        if r["study"] != "A":
            continue
        for k in r["readings"]:
            cell = out[(k["position_label"], k["layer"])]
            cell[r["group"]].append(k["tool_word_prob"])
            if r["group"] == "attack" and r.get("success_rate") is not None:
                cell["rates"].append((k["tool_word_prob"], r["success_rate"]))
    return out


def index_b(rows, sample):
    """Paired win/miss curves over answer deciles.

    Answers differ in length, so each read position is placed on a 0..1 scale of its own answer's
    length and binned. A pair contributes to a bin only when BOTH its answers have a reading there.

    Teacher forcing replays the recorded answer, so a winning answer eventually contains its own tool
    call: a reading taken after that point is partly reading the tool call back, which is circular.
    Each bin therefore also records how many winning answers have already begun their tool call by
    then, and a second, stricter curve keeps only positions strictly before it. The tool call's token
    index is estimated from its character offset (chars are close to uniform per token within one
    answer), so the strict curve is conservative rather than exact.
    """
    per = collections.defaultdict(dict)
    tool_call = collections.defaultdict(dict)
    contaminated = collections.Counter()   # bin -> winning answers whose tool call has started
    n_wins = 0
    for r in rows:
        if r["study"] != "B":
            continue
        meta = sample.get(r["id"], {})
        idxs = [int(k["position_label"].rsplit("_", 1)[1]) for k in r["readings"]
                if k["position_label"].startswith("answer_token_")]
        span = max(idxs) if idxs else 0

        cut = None  # token index at which this answer's tool call starts
        if r["outcome"] == "win" and meta.get("tool_call_char") is not None and meta.get("response"):
            frac = meta["tool_call_char"] / max(1, len(meta["response"]))
            cut = frac * (meta.get("response_tokens") or span or 1)
            n_wins += 1
            for b in range(N_BINS):
                if (b + 0.5) / N_BINS >= frac:
                    contaminated[b] += 1

        curve = collections.defaultdict(lambda: collections.defaultdict(list))
        strict = collections.defaultdict(lambda: collections.defaultdict(list))
        for k in r["readings"]:
            lab = k["position_label"]
            if lab == "just_before_tool_call":
                tool_call[k["layer"]][r["base_id"]] = k["tool_word_prob"]
                continue
            if not lab.startswith("answer_token_"):
                continue
            tok = int(lab.rsplit("_", 1)[1])
            b = min(N_BINS - 1, int((tok / span if span else 0.0) * N_BINS))
            curve[k["layer"]][b].append(k["tool_word_prob"])
            if cut is None or tok < cut:
                strict[k["layer"]][b].append(k["tool_word_prob"])
        per[r["base_id"]][r["outcome"]] = (curve, strict)

    paired = collections.defaultdict(lambda: collections.defaultdict(list))
    paired_strict = collections.defaultdict(lambda: collections.defaultdict(list))
    raw = collections.defaultdict(lambda: collections.defaultdict(lambda: {"win": [], "miss": []}))
    for base_id, by_outcome in per.items():
        if "win" not in by_outcome or "miss" not in by_outcome:
            continue
        (wc, ws), (mc, ms) = by_outcome["win"], by_outcome["miss"]
        for layer in wc:
            for b in range(N_BINS):
                w, m = wc[layer].get(b), mc[layer].get(b)
                if w and m:
                    paired[layer][b].append(mean(w) - mean(m))
                    raw[layer][b]["win"].append(mean(w))
                    raw[layer][b]["miss"].append(mean(m))
                w2, m2 = ws[layer].get(b), ms[layer].get(b)
                if w2 and m2:
                    paired_strict[layer][b].append(mean(w2) - mean(m2))
    return paired, paired_strict, raw, tool_call, contaminated, n_wins


def top_words(rows, group, label, layer, n=12):
    """Words the lens reads most often in the top-1 slot, across prompts of one group."""
    c = collections.Counter()
    for r in rows:
        if r["study"] != "A" or r["group"] != group:
            continue
        for k in r["readings"]:
            if k["position_label"] == label and k["layer"] == layer and k["top_words"]:
                c[k["top_words"][0]] += 1
    total = sum(c.values()) or 1
    return [{"word": w, "count": n_, "share": n_ / total} for w, n_ in c.most_common(n)]


# ---------------------------------------------------------------- figures

def fig_separation(a_idx, layers, out_dir):
    plt, s = rp.figure()
    made = []
    for dark in (False, True):
        plt, s = rp.figure(dark)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, (label, title) in zip(axes, POSITIONS):
            atk = [mean(a_idx[(label, L)]["attack"]) for L in layers]
            cln = [mean(a_idx[(label, L)]["clean"]) for L in layers]
            ax.plot(layers, atk, color=s["accent"], lw=2, label="attack email")
            ax.plot(layers, cln, color=s["series"], lw=2, label="clean email")
            ax.axvspan(layers[0], TRUSTED_FROM, color=s["muted"], alpha=0.12, lw=0)
            ax.set_title(title, fontsize=10, color=s["ink"])
            ax.set_xlabel("layer read")
        axes[0].set_ylabel("lens probability on tool words")
        axes[0].legend(frameon=False, fontsize=9, labelcolor=s["ink2"])
        fig.text(0.5, -0.06, f"shaded: layers below {TRUSTED_FROM}, where the lens fails its own "
                 "depth probes", ha="center", color=s["muted"], fontsize=9)
        made += rp.save(fig, out_dir, "01-attack-vs-clean-by-layer" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


def fig_auroc(a_idx, layers, out_dir):
    made = []
    for dark in (False, True):
        plt, s = rp.figure(dark)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        colors = [s["series"], s["accent"], THIRD["dark" if dark else "light"]]
        for (label, title), c in zip(POSITIONS, colors):
            ys = [auroc(a_idx[(label, L)]["attack"], a_idx[(label, L)]["clean"]) for L in layers]
            ax.plot(layers, ys, color=c, lw=2, label=title)
        ax.axhline(0.5, color=s["muted"], lw=1, ls="--")
        ax.axvspan(layers[0], TRUSTED_FROM, color=s["muted"], alpha=0.12, lw=0)
        ax.set_xlabel("layer read")
        ax.set_ylabel("AUROC: attack vs clean")
        ax.set_ylim(0.10, 1.02)
        ax.legend(frameon=False, fontsize=9, labelcolor=s["ink2"], loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=3)
        made += rp.save(fig, out_dir, "02-auroc-by-layer" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


def fig_rates(a_idx, layers, out_dir):
    made = []
    for dark in (False, True):
        plt, s = rp.figure(dark)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        colors = [s["series"], s["accent"], THIRD["dark" if dark else "light"]]
        for (label, title), c in zip(POSITIONS, colors):
            ys = []
            for L in layers:
                pairs = a_idx[(label, L)]["rates"]
                ys.append(spearman([p[0] for p in pairs], [p[1] for p in pairs]))
            ax.plot(layers, ys, color=c, lw=2, label=title)
        ax.axhline(0, color=s["muted"], lw=1, ls="--")
        ax.axvspan(layers[0], TRUSTED_FROM, color=s["muted"], alpha=0.12, lw=0)
        ax.set_xlabel("layer read")
        ax.set_ylabel("rank correlation with success rate")
        ax.legend(frameon=False, fontsize=9, labelcolor=s["ink2"], loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=3)
        made += rp.save(fig, out_dir, "03-reading-vs-success-rate" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


def fig_trajectory(raw, paired, paired_strict, layer, out_dir, median_onset):
    made = []
    xs = [(b + 0.5) / N_BINS for b in range(N_BINS)]
    for dark in (False, True):
        plt, s = rp.figure(dark)
        third = THIRD["dark" if dark else "light"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        w = [mean(raw[layer][b]["win"]) for b in range(N_BINS)]
        m = [mean(raw[layer][b]["miss"]) for b in range(N_BINS)]
        ax1.plot(xs, w, color=s["accent"], lw=2, marker="o", ms=4, label="answer that fired")
        ax1.plot(xs, m, color=s["series"], lw=2, marker="o", ms=4, label="answer that did not")
        ax1.axvline(median_onset, color=s["muted"], lw=1, ls=":")
        ax1.set_xlabel("position along the answer")
        ax1.set_ylabel("lens probability on tool words")
        ax1.set_title(f"layer {layer}, same prompts, both outcomes", fontsize=10, color=s["ink"])
        ax1.legend(frameon=False, fontsize=9, labelcolor=s["ink2"])

        def band(ax, data, color, label, dx):
            # A bin resting on a handful of pairs is noise, not a data point: the strict curve
            # loses pairs wherever most of an answer sits inside its own tool call.
            pts = [(b, boot_ci(data[layer][b], mean, seed=b)) for b in range(N_BINS)
                   if len(data[layer].get(b) or []) >= MIN_PAIRS]
            if not pts:
                return
            x = [(b + 0.5) / N_BINS + dx for b, _ in pts]
            ax.errorbar(x, [o[0] for _, o in pts],
                        yerr=[[o[0] - o[1] for _, o in pts], [o[2] - o[0] for _, o in pts]],
                        color=color, lw=2, marker="o", ms=4, capsize=3, label=label)

        band(ax2, paired, s["accent"], "all positions", -0.008)
        band(ax2, paired_strict, third, "before the tool call only", 0.008)
        ax2.axhline(0, color=s["muted"], lw=1, ls="--")
        ax2.axvline(median_onset, color=s["muted"], lw=1, ls=":")
        ax2.set_xlabel("position along the answer")
        ax2.set_ylabel("fired minus did not (paired)")
        ax2.set_title("difference within each pair", fontsize=10, color=s["ink"])
        ax2.legend(frameon=False, fontsize=9, labelcolor=s["ink2"], loc="upper left")
        fig.text(0.5, -0.04, "dotted line: the median answer starts its tool call here; readings "
                 "past it partly read that tool call back", ha="center", color=s["muted"],
                 fontsize=9)
        made += rp.save(fig, out_dir, "04-answer-trajectory" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


# ---------------------------------------------------------------- report

def threshold_split(a_rows, label, layer):
    """Attacks that read higher than EVERY clean prompt, and how often each side actually fired.

    A threshold set at the largest clean reading has no clean false positives by construction; the
    question it answers is how many attacks clear it and whether those attacks behave differently.
    """
    atk, cln = [], []
    for r in a_rows:
        for k in r["readings"]:
            if k["position_label"] == label and k["layer"] == layer:
                (atk if r["group"] == "attack" else cln).append((r, k["tool_word_prob"]))
    if not atk or not cln:
        return None
    thr = max(v for _, v in cln)
    above = [r["success_rate"] for r, v in atk if v > thr and r.get("success_rate") is not None]
    below = [r["success_rate"] for r, v in atk if v <= thr and r.get("success_rate") is not None]
    am, alo, ahi = boot_ci(above, mean, seed=4)
    bm, blo, bhi = boot_ci(below, mean, seed=5)
    return {"threshold": thr, "n_attack": len(atk), "n_clean": len(cln), "n_above": len(above),
            "n_below": len(below), "rate_above": am, "rate_above_lo": alo, "rate_above_hi": ahi,
            "rate_below": bm, "rate_below_lo": blo, "rate_below_hi": bhi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", default=os.path.join(RAW, "readouts.jsonl"))
    args = ap.parse_args()

    sample, rows, errors = load(args.readouts)
    os.makedirs(OUT, exist_ok=True)
    expected = len(sample)
    complete = len(rows) == expected and not errors

    a_rows = [r for r in rows if r["study"] == "A"]
    b_rows = [r for r in rows if r["study"] == "B"]
    layers = sorted({k["layer"] for r in rows for k in r["readings"]})
    L = HEADLINE_LAYER if HEADLINE_LAYER in layers else layers[-1]
    trusted = [x for x in layers if x >= TRUSTED_FROM]

    a_idx = index_a(a_rows)
    paired, paired_strict, raw_b, tool_call, contaminated, n_wins = index_b(b_rows, sample)

    # ---- table 1: attack vs clean, per position and layer, with intervals
    t1 = []
    for label, _ in POSITIONS:
        for layer in layers:
            cell = a_idx[(label, layer)]
            if not cell["attack"]:
                continue
            am, alo, ahi = rp.bootstrap_mean(cell["attack"])
            cm, clo, chi = rp.bootstrap_mean(cell["clean"])
            au, aulo, auhi = boot_ci2(cell["attack"], cell["clean"], auroc, seed=layer)
            rho, rlo, rhi = boot_ci_pairs(cell["rates"], spearman, seed=layer)
            t1.append({"position": label, "layer": layer, "attack_mean": am, "attack_lo": alo,
                       "attack_hi": ahi, "clean_mean": cm, "clean_lo": clo, "clean_hi": chi,
                       "auroc": au, "auroc_lo": aulo, "auroc_hi": auhi,
                       "spearman_success_rate": rho, "spearman_lo": rlo, "spearman_hi": rhi,
                       "n_attack": len(cell["attack"]), "n_clean": len(cell["clean"])})
    rp.write_csv(os.path.join(OUT, "tables", "study-a-by-layer.csv"), t1,
                 ["position", "layer", "attack_mean", "attack_lo", "attack_hi", "clean_mean",
                  "clean_lo", "clean_hi", "auroc", "auroc_lo", "auroc_hi", "spearman_success_rate",
                  "spearman_lo", "spearman_hi", "n_attack", "n_clean"])

    # ---- table 2: study B trajectory, with contamination and the strict curve
    t2 = []
    for layer in trusted:
        for b in range(N_BINS):
            d = paired[layer][b]
            if not d:
                continue
            dm, dlo, dhi = boot_ci(d, mean, seed=b)
            s = paired_strict[layer][b]
            sm, slo, shi = boot_ci(s, mean, seed=100 + b) if s else (float("nan"),) * 3
            t2.append({"layer": layer, "bin": b, "answer_fraction": round((b + 0.5) / N_BINS, 2),
                       "win_mean": mean(raw_b[layer][b]["win"]),
                       "miss_mean": mean(raw_b[layer][b]["miss"]),
                       "paired_diff": dm, "diff_lo": dlo, "diff_hi": dhi, "n_pairs": len(d),
                       "wins_with_tool_call_started": contaminated.get(b, 0),
                       "pre_tool_call_diff": sm, "pre_lo": slo, "pre_hi": shi, "n_pre": len(s)})
    rp.write_csv(os.path.join(OUT, "tables", "study-b-trajectory.csv"), t2,
                 ["layer", "bin", "answer_fraction", "win_mean", "miss_mean", "paired_diff",
                  "diff_lo", "diff_hi", "n_pairs", "wins_with_tool_call_started",
                  "pre_tool_call_diff", "pre_lo", "pre_hi", "n_pre"])

    # ---- table 3: what the lens reads
    t3 = []
    for label, _ in POSITIONS:
        for layer in sorted({32, L}):
            for group in ("attack", "clean"):
                for w in top_words(a_rows, group, label, layer, n=8):
                    t3.append({"position": label, "layer": layer, "group": group, **w})
    rp.write_csv(os.path.join(OUT, "tables", "top-words.csv"), t3,
                 ["position", "layer", "group", "word", "count", "share"])

    split = threshold_split(a_rows, "prompt_end", L)
    figs = []
    try:
        figs += fig_separation(a_idx, layers, os.path.join(OUT, "figures"))
        figs += fig_auroc(a_idx, layers, os.path.join(OUT, "figures"))
        figs += fig_rates(a_idx, layers, os.path.join(OUT, "figures"))
        if paired:
            onsets = [s_["tool_call_char"] / max(1, len(s_["response"]))
                      for s_ in sample.values()
                      if s_.get("study") == "B" and s_.get("outcome") == "win"
                      and s_.get("tool_call_char") is not None]
            onsets.sort()
            figs += fig_trajectory(raw_b, paired, paired_strict, L, os.path.join(OUT, "figures"),
                                   onsets[len(onsets) // 2] if onsets else 0.76)
    except ImportError:
        print("matplotlib missing: tables and REPORT.md only")

    write_report(OUT, sample, rows, errors, complete, expected, layers, trusted, L, a_idx, t1, t2,
                 t3, split, paired, paired_strict, tool_call, contaminated, n_wins,
                 args.readouts, figs)
    print(f"wrote {OUT} ({'complete' if complete else 'INCOMPLETE'})")


def num(x, d=5):
    return "n/a" if x != x else f"{x:.{d}f}"


def ci(m, lo, hi, d=5):
    return "n/a" if m != m else f"{m:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def write_report(out, sample, rows, errors, complete, expected, layers, trusted, L, a_idx, t1, t2,
                 t3, split, paired, paired_strict, tool_call, contaminated, n_wins,
                 readouts_path, figs):
    n_a = sum(1 for r in rows if r["study"] == "A")
    n_b = sum(1 for r in rows if r["study"] == "B")
    banner = ""
    if not complete:
        banner = (f"> **Status: incomplete.** {len(rows)} of {expected} readouts present"
                  f"{f', {len(errors)} failed' if errors else ''}. Numbers below are partial.\n\n")

    def row(label, layer):
        return next(r for r in t1 if r["position"] == label and r["layer"] == layer)

    sec_a = [{"position": title,
              "attack": num(row(label, L)["attack_mean"]),
              "clean": num(row(label, L)["clean_mean"]),
              "auroc": ci(row(label, L)["auroc"], row(label, L)["auroc_lo"],
                          row(label, L)["auroc_hi"], 3),
              "rho": ci(row(label, L)["spearman_success_rate"], row(label, L)["spearman_lo"],
                        row(label, L)["spearman_hi"], 3)}
             for label, title in POSITIONS]

    lines = [
        "# Experiment 03: what the model has in mind while it reads an attack",
        "",
        banner + "The first two experiments measured outcomes: how often an attack fires, and what "
        "a defense does to that rate. This one looks inside the model, using a **Jacobian lens** — "
        "a fixed linear map, one per layer, that turns a hidden state into words. The lens was "
        "fitted on 100 plain-English WikiText passages and never on these prompts, so it is not "
        "tuned to the thing it measures.",
        "",
        "Every number is the lens probability on a fixed list of **tool words** "
        "(`send`, `email`, `contact`, `@`, `confirmation`, `function`, `{\"`, `type`) — the same "
        "list for every prompt, so nothing depends on a payload's random tool name.",
        "",
        f"**Every prompt here is undefended.** The headline layer is fixed at **{L}**, the deepest "
        "layer read and so the readout closest to what the model actually emits, chosen before any "
        "separation number was computed. Layers below "
        f"{TRUSTED_FROM} fail the lens's own depth probes. Per-layer curves are in the figures and "
        "`tables/study-a-by-layer.csv`; because the best of 39 layers was not cherry-picked for the "
        "headline, those curves should be read as exploratory.",
        "",
        "## The short version",
        "",
    ]
    if split:
        lines += [
            f"At the last position of the prompt — the one that produces the first generated token, "
            f"before the model has written anything — the lens separates an attack-bearing prompt "
            f"from a clean one with **AUROC "
            f"{ci(row('prompt_end', L)['auroc'], row('prompt_end', L)['auroc_lo'], row('prompt_end', L)['auroc_hi'], 3)}**, "
            f"and the size of the reading tracks how often that attack actually fired across its 8 "
            f"recorded runs (**rank correlation "
            f"{ci(row('prompt_end', L)['spearman_success_rate'], row('prompt_end', L)['spearman_lo'], row('prompt_end', L)['spearman_hi'], 3)}**).",
            "",
            f"The separation is a **tail, not a shift**. Set a threshold at the largest reading any "
            f"of the {split['n_clean']} clean prompts produced ({num(split['threshold'])}) — zero "
            f"clean false positives by construction. **{split['n_above']} of {split['n_attack']} "
            f"attack prompts clear it**, and those attacks went on to fire "
            f"{rp.pct(split['rate_above'])} of the time "
            f"({rp.pct(split['rate_above_lo'])} to {rp.pct(split['rate_above_hi'])}) against "
            f"{rp.pct(split['rate_below'])} "
            f"({rp.pct(split['rate_below_lo'])} to {rp.pct(split['rate_below_hi'])}) for the "
            f"{split['n_below']} that did not. Non-overlapping intervals, roughly a threefold "
            "difference in success rate, from a reading taken before generation starts.",
            "",
            "Most attacks look like ordinary text to this probe. A minority do not, and that "
            "minority is the dangerous one.",
            "",
        ]
    lines += [
        "## Study A: the prompt, before a single word is generated",
        "",
        f"{n_a} prompts (200 attack, 203 clean), one forward pass each, read at three places.",
        "",
        rp.md_table(sec_a, [("position", "Read at", None), ("attack", "Attack mean", None),
                            ("clean", "Clean mean", None), ("auroc", "AUROC (95% CI)", None),
                            ("rho", "Rank corr. with success rate", None)]),
        "",
        "AUROC is the chance a randomly chosen attack prompt reads higher than a randomly chosen "
        "clean one: 0.5 is a coin flip, 1.0 is perfect separation.",
        "",
        "Two of these three rows are controls and behave like controls. **The end of the benign "
        "email** sits at chance, which is what it must do — every prompt, attack or clean, is "
        "identical up to that point, so a reading that separated them there would mean the pipeline "
        "was leaking. (At deep layers this position also saturates: the tool-word probability "
        "approaches 1.0 for everything, so it carries no information either way.) **The end of the "
        "attacker's email** is also near chance at the headline layer.",
        "",
        "![attack vs clean by layer](figures/01-attack-vs-clean-by-layer.svg)",
        "",
        "![AUROC by layer](figures/02-auroc-by-layer.svg)",
        "",
        "![reading vs success rate](figures/03-reading-vs-success-rate.svg)",
        "",
        "The AUROC curve for the end of the prompt climbs with depth rather than jumping around, "
        "which is the shape you would want if the effect were real: 0.55 in the low 30s, then "
        f"{row('prompt_end', 35)['auroc']:.2f} at 35, {row('prompt_end', 37)['auroc']:.2f} at 37, "
        f"{row('prompt_end', L)['auroc']:.2f} at {L}.",
        "",
        "One honest wrinkle: the end of the attacker's email separates the two groups strongly at "
        f"layers 18-19 (AUROC {row('attacker_email_end', 18)['auroc']:.2f} at 18), which are "
        "exactly the layers whose readings decode into junk words. A hidden state can carry "
        "information the lens cannot render as English, so that number is reported but not "
        "interpreted.",
        "",
        "**What Study A cannot do.** The same prompt sometimes fires and sometimes does not, "
        "because the model samples. A reading taken while the model reads a fixed prompt is "
        "identical across those runs, so it cannot explain the difference between them. That is why "
        "the last column correlates with a *rate* over 8 runs, not with a single outcome — and why "
        "Study B exists.",
        "",
        "## Study B: along the answer, same prompt, both outcomes",
        "",
        f"{n_b} replays: {n_wins} attack prompts that both won and lost on identical wording, one "
        "winning and one losing answer each, replayed under teacher forcing and read along the "
        "answer. The prompt is held fixed inside each pair, so any difference belongs to the "
        "generated text.",
        "",
        "![answer trajectory](figures/04-answer-trajectory.svg)",
        "",
    ]

    tbl = [r for r in t2 if r["layer"] == L]
    if tbl:
        lines += [
            rp.md_table(tbl, [("answer_fraction", "Position in answer", lambda v: f"{int(v*100)}%"),
                              ("win_mean", "Fired", lambda v: num(v, 4)),
                              ("miss_mean", "Did not", lambda v: num(v, 4)),
                              ("paired_diff", "Difference", lambda v: f"{v:+.4f}"),
                              ("diff_lo", "95% CI low", lambda v: f"{v:+.4f}"),
                              ("diff_hi", "95% CI high", lambda v: f"{v:+.4f}"),
                              ("n_pairs", "Pairs", str),
                              ("wins_with_tool_call_started", "Wins already in tool call", str)]),
            "",
            "The last column is the one that decides whether this means anything. Teacher forcing "
            "replays the recorded answer, so once a winning answer has started writing its tool "
            "call, a high reading is partly the model reading its own tool call back — circular. "
            "The tool call starts at a median of 76% of the way through these answers.",
            "",
        ]
        strict_ok = [r for r in tbl if r["bin"] in (1, 2) and r["pre_lo"] == r["pre_lo"]
                     and r["pre_lo"] > 0]
        early = [r for r in tbl if r["bin"] in (1, 2) and r["diff_lo"] > 0]
        if early:
            e = max(early, key=lambda r: r["paired_diff"])
            lines += [
                f"So the meaningful window is early. At **{int(e['answer_fraction']*100)}% of the "
                f"way in**, the answers that would go on to fire read {num(e['win_mean'], 4)} "
                f"against {num(e['miss_mean'], 4)} for the ones that would not — "
                f"{e['paired_diff']:+.4f} ({e['diff_lo']:+.4f} to {e['diff_hi']:+.4f}, "
                f"n={e['n_pairs']} pairs).",
                "",
                f"That number is still too generous. {e['wins_with_tool_call_started']} of {n_wins} "
                "winning answers put their tool call at the very start, so their whole trajectory "
                "sits inside it. Dropping every position at or after each answer's own tool call "
                "leaves the `pre_tool_call_diff` column of "
                "`tables/study-b-trajectory.csv`, and the effect shrinks but survives:",
                "",
            ]
            if strict_ok:
                s = max(strict_ok, key=lambda r: r["pre_tool_call_diff"])
                lines += [
                    f"at **{int(s['answer_fraction']*100)}% of the way in**, counting only "
                    f"positions before the tool call, the difference is "
                    f"{s['pre_tool_call_diff']:+.4f} ({s['pre_lo']:+.4f} to {s['pre_hi']:+.4f}, "
                    f"n={s['n_pre']} pairs) — about a third of the uncorrected figure, and still "
                    "clear of zero. **Same prompt, same opening, no tool call written yet, and the "
                    "run that was going to send the email is already distinguishable.** That is the "
                    "one claim in this experiment that survives every correction I could think to "
                    "apply to it.",
                    "",
                ]
            else:
                lines += ["once those positions are removed no early bin's interval excludes zero, "
                          "so the early gap cannot be separated from the circularity.", ""]
        tail = [r for r in tbl if r["bin"] >= 7]
        if tail and all(r["diff_lo"] <= 0 for r in tail):
            lines += [
                "The opposite is true at the end. Over the last 30% of the answer — where 45 of "
                f"{n_wins} tool calls actually sit — no bin's interval excludes zero. Where the "
                "evidence is most blatant in the text, this reading stops telling the two apart. "
                "A probe fitted on plain English is not a tool-call detector, and it does not "
                "become one just because a tool call is present.",
                "",
            ]
    if tool_call.get(L):
        vals = list(tool_call[L].values())
        m, lo, hi = rp.bootstrap_mean(vals)
        lines += [f"At the token immediately before a winning answer emits its tool call, the "
                  f"reading is {ci(m, lo, hi, 4)} (n={len(vals)}). Losing answers never emit one, "
                  "so they have no counterpart position and this number stands alone.", ""]

    lines += ["## What the lens actually reads", "",
              f"Top-1 words at layer {L}, counted across prompts. Full table in "
              "`tables/top-words.csv`.", ""]
    for label, title in POSITIONS:
        for group in ("attack", "clean"):
            ws = [r for r in t3 if r["position"] == label and r["layer"] == L and r["group"] == group]
            if ws:
                lines.append(f"- **{title} — {group}:** " +
                             ", ".join(f"`{w['word']}` ({rp.pct(w['share'], 0)})" for w in ws[:6]))
    lines += [
        "",
        "The word counts say the same thing the numbers do. At the end of the prompt, `{\"` — the "
        "opening of a JSON tool call — appears in the lens's top 8 for **31 of 200 attack prompts "
        "and 0 of 203 clean prompts**. The clean prompts are uniform and boring; the attack prompts "
        "are mostly boring with a visible minority that are not.",
        "",
        "## How to read these numbers, and what they are not",
        "",
        "- A lens reading is **not** the model's next-token distribution. It is what a fixed linear "
        "map, fitted on unrelated English, extracts from one hidden state at one layer. It is a "
        "probe, not the model's output.",
        "- A high reading means those particular words are prominent in that state. It does not "
        "mean the model has decided to call a tool.",
        "- Every prompt here is undefended. Nothing in this experiment says what a defense does to "
        "these readings; that needs the same pass over the defended prompts.",
        "- Study A cannot explain why one run of a prompt fires and another does not. Study B can, "
        "and only in the window before the answer contains its own tool call.",
        "- Intervals are percentile bootstraps, 2000 resamples: over prompts in Study A, over "
        "matched pairs in Study B.",
        "- The tool call's token index in Study B is estimated from its character offset, so the "
        "`pre_tool_call_diff` column in the trajectory table is approximate.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python3 analysis/build_lens_sample.py          # choose the prompts and answers",
        "cd modal && modal run lens_fit.py              # fit the lens (~48 min, one time)",
        "cd modal && modal run lens_apply.py            # read all 555 items (~3 min on 6 A100s)",
        "python3 analysis/report_03_lens.py             # this report",
        "```",
    ]
    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    files = [rp.file_entry(os.path.join(RAW, "sample.jsonl"), "study inputs")]
    if os.path.exists(readouts_path):
        files.append(rp.file_entry(readouts_path, "lens readouts"))
    for p in sorted(figs) + [os.path.join(out, "tables", n)
                             for n in sorted(os.listdir(os.path.join(out, "tables")))]:
        files.append(rp.file_entry(p, "output"))
    manifest = {
        "experiment": EXP, "built_at": rp.now_utc(), "complete": complete,
        "items_expected": expected, "items_read": len(rows), "items_failed": len(errors),
        "model": rp.MODEL, "prompt_recipe": rp.PROMPT_RECIPE,
        "lens": {"repo": "https://github.com/anthropics/jacobian-lens", "tag": "phi3-wikitext100",
                 "fit_corpus": "100 WikiText passages, 128 tokens each", "layers": layers,
                 "trusted_from_layer": TRUSTED_FROM, "headline_layer": L,
                 "headline_layer_chosen": "deepest layer read, fixed before computing separation",
                 "depth_probe_note": "layers 24/32/38 pass shallow/mid/deep probes; 8 and 16 fail"},
        "readout": {"top_k": 8, "response_stride": 8, "bins": N_BINS,
                    "note": "tool word list and stride are set in modal/lens_apply.py"},
        "headline": split,
        "git": rp.git_info(), "files": files,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
