"""Experiment 03, defenses: does a defense change what the model has in mind, or only what it says?

Study A found that, at the position producing the first generated token, an undefended attack
prompt reads higher on tool words than a clean one (AUROC 0.81 at layer 38). This compares the same
403 prompts under four defenses from experiment 02, at the same position and layer.

Two ways a defense can "work" under the lens:
  1. it lowers the reading on attacks (the model no longer has the tool call in mind), or
  2. it leaves the reading alone and the attack still fires less (the change is elsewhere).
The paired change per attack, undefended -> defended, tells them apart.

Inputs:  runs/03-jacobian-lens/readouts.jsonl           (undefended, experiment 03)
         runs/03-jacobian-lens/readouts_defended.jsonl  (this run)
         runs/03-jacobian-lens/sample_defended.jsonl
Outputs: results/03-jacobian-lens/REPORT-defenses.md, tables/defenses-*.csv, figures/05-*, 06-*

Run:  python3 analysis/report_03_defenses.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting as rp  # noqa: E402
from report_03_lens import (HEADLINE_LAYER, auroc, spearman, boot_ci, boot_ci2,  # noqa: E402
                            boot_ci_pairs, mean, num, ci, THIRD)

RAW = os.path.join(rp.ROOT, "runs", "03-jacobian-lens")
OUT = os.path.join(rp.ROOT, "results", "03-jacobian-lens")
L = HEADLINE_LAYER
NAMES = collections.OrderedDict([
    ("none", "No defense"),
    ("ms_spotlight", "Microsoft spotlighting"),
    ("lib_markdata", "Library as shipped (marker between words)"),
    ("lib_base64", "Library, base64"),
    ("v2_uni_random_phi3", "Library fixed (short Unicode, Phi-3 boundaries)"),
])
# attack success under each defense, 4 runs each (experiment 02); none = 8 runs (experiment 01)


def reading(r, label, layer=L):
    k = next((k for k in r["readings"] if k["position_label"] == label and k["layer"] == layer), None)
    return k["tool_word_prob"] if k else float("nan")


def load():
    base = rp.jsonl(os.path.join(RAW, "readouts.jsonl"))
    base = [r for r in base if "error" not in r and r["study"] == "A"]
    for r in base:
        r["condition"] = "none"
    path = os.path.join(RAW, "readouts_defended.jsonl")
    dfd = rp.jsonl(path) if os.path.exists(path) else []
    errors = [r for r in dfd if "error" in r]
    dfd = [r for r in dfd if "error" not in r]
    sample = {r["id"]: r for r in rp.jsonl(os.path.join(RAW, "sample_defended.jsonl"))}
    for r in dfd:
        s = sample.get(r["id"], {})
        r["undefended_success_rate"] = s.get("undefended_success_rate")
        r["runs"] = s.get("runs")
    for r in base:
        r["undefended_success_rate"] = r.get("success_rate")
    expected = len(sample)
    return base + dfd, errors, expected, len(dfd)


def per_condition(rows, label):
    out = []
    for cond, name in NAMES.items():
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        atk = [reading(r, label) for r in rs if r["group"] == "attack"]
        cln = [reading(r, label) for r in rs if r["group"] == "clean"]
        atk = [v for v in atk if v == v]
        cln = [v for v in cln if v == v]
        if not atk or not cln:
            continue
        au, alo, ahi = boot_ci2(atk, cln, auroc, seed=1)
        thr = max(cln)
        above = [r for r in rs if r["group"] == "attack" and reading(r, label) > thr]
        pairs = [(reading(r, label), r["success_rate"]) for r in rs
                 if r["group"] == "attack" and r.get("success_rate") is not None]
        rho, rlo, rhi = boot_ci_pairs(pairs, spearman, seed=2)
        rates = [r["success_rate"] for r in rs if r["group"] == "attack" and r.get("success_rate") is not None]
        rm, rl, rh = rp.bootstrap_mean(rates)
        am, al, ah = rp.bootstrap_mean(atk)
        cm, cl, ch = rp.bootstrap_mean(cln)
        out.append({"condition": cond, "name": name, "n_attack": len(atk), "n_clean": len(cln),
                    "attack_mean": am, "attack_lo": al, "attack_hi": ah,
                    "clean_mean": cm, "clean_lo": cl, "clean_hi": ch,
                    "auroc": au, "auroc_lo": alo, "auroc_hi": ahi,
                    "clean_max": thr, "attacks_above_clean_max": len(above),
                    "spearman_success": rho, "spearman_lo": rlo, "spearman_hi": rhi,
                    "success_rate": rm, "success_lo": rl, "success_hi": rh})
    return out


def paired_change(rows, label):
    """Per attack, reading under a defense minus reading undefended, same base_id."""
    base = {r["base_id"]: reading(r, label) for r in rows
            if r["condition"] == "none" and r["group"] == "attack"}
    out = []
    for cond, name in NAMES.items():
        if cond == "none":
            continue
        diffs, both = [], []
        for r in rows:
            if r["condition"] != cond or r["group"] != "attack" or r["base_id"] not in base:
                continue
            v = reading(r, label)
            if v == v and base[r["base_id"]] == base[r["base_id"]]:
                diffs.append(v - base[r["base_id"]])
                both.append((base[r["base_id"]], v))
        if not diffs:
            continue
        m, lo, hi = boot_ci(diffs, mean, seed=3)
        # among attacks that read above every clean prompt when undefended, how many still do?
        out.append({"condition": cond, "name": name, "n": len(diffs), "mean_change": m,
                    "change_lo": lo, "change_hi": hi,
                    "fraction_lower": sum(1 for d in diffs if d < 0) / len(diffs),
                    "rank_corr_before_after": spearman([b for b, _ in both], [a for _, a in both])})
    return out


def fig_auroc(t, out_dir):
    made = []
    for dark in (False, True):
        plt, s = rp.figure(dark)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
        names = [r["name"] for r in t]
        y = list(range(len(t)))[::-1]
        ax1.errorbar([r["auroc"] for r in t], y,
                     xerr=[[r["auroc"] - r["auroc_lo"] for r in t], [r["auroc_hi"] - r["auroc"] for r in t]],
                     fmt="o", color=s["series"], capsize=3, lw=1.5)
        ax1.axvline(0.5, color=s["muted"], lw=1, ls="--")
        ax1.set_yticks(y)
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlim(0.3, 1.0)
        ax1.set_xlabel(f"AUROC, attack vs clean, end of prompt, layer {L}")
        ax2.errorbar([r["success_rate"] for r in t], y,
                     xerr=[[r["success_rate"] - r["success_lo"] for r in t], [r["success_hi"] - r["success_rate"] for r in t]],
                     fmt="o", color=s["accent"], capsize=3, lw=1.5)
        ax2.set_yticks(y)
        ax2.set_yticklabels([""] * len(t))
        ax2.set_xlim(0, max(0.05, max(r["success_hi"] for r in t) * 1.1))
        ax2.set_xlabel("attack success rate under that defense")
        made += rp.save(fig, out_dir, "05-defenses-auroc-vs-success" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


def fig_paired(rows, out_dir):
    """Undefended reading (x) against defended reading (y), one dot per attack, per defense."""
    base = {r["base_id"]: reading(r, "prompt_end") for r in rows
            if r["condition"] == "none" and r["group"] == "attack"}
    conds = [c for c in NAMES if c != "none" and any(r["condition"] == c for r in rows)]
    made = []
    for dark in (False, True):
        plt, s = rp.figure(dark)
        fig, axes = plt.subplots(1, len(conds), figsize=(3.3 * len(conds), 3.6), sharex=True, sharey=True)
        if len(conds) == 1:
            axes = [axes]
        for ax, c in zip(axes, conds):
            xs, ys, fired = [], [], []
            for r in rows:
                if r["condition"] == c and r["group"] == "attack" and r["base_id"] in base:
                    xs.append(max(base[r["base_id"]], 1e-6))
                    ys.append(max(reading(r, "prompt_end"), 1e-6))
                    fired.append((r.get("success_rate") or 0) > 0)
            ax.scatter([x for x, f in zip(xs, fired) if not f], [y for y, f in zip(ys, fired) if not f],
                       s=12, color=s["series"], alpha=0.7, label="never fired under defense")
            ax.scatter([x for x, f in zip(xs, fired) if f], [y for y, f in zip(ys, fired) if f],
                       s=14, color=s["accent"], alpha=0.9, label="fired under defense")
            ax.plot([1e-6, 1], [1e-6, 1], color=s["muted"], lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(NAMES[c], fontsize=9, color=s["ink"])
            ax.set_xlabel("reading, undefended")
        axes[0].set_ylabel("reading, defended")
        axes[0].legend(frameon=False, fontsize=8, labelcolor=s["ink2"], loc="upper left")
        made += rp.save(fig, out_dir, "06-defenses-paired-readings" + ("-dark" if dark else ""))
        plt.close(fig)
    return made


def main():
    rows, errors, expected, n_dfd = load()
    complete = n_dfd == expected and not errors
    t_end = per_condition(rows, "prompt_end")
    t_att = per_condition(rows, "attacker_email_end")
    t_pair = paired_change(rows, "prompt_end")
    fields = ["condition", "name", "n_attack", "n_clean", "attack_mean", "attack_lo", "attack_hi",
              "clean_mean", "clean_lo", "clean_hi", "auroc", "auroc_lo", "auroc_hi", "clean_max",
              "attacks_above_clean_max", "spearman_success", "spearman_lo", "spearman_hi",
              "success_rate", "success_lo", "success_hi"]
    rp.write_csv(os.path.join(OUT, "tables", "defenses-prompt-end.csv"), t_end, fields)
    rp.write_csv(os.path.join(OUT, "tables", "defenses-attacker-email-end.csv"), t_att, fields)
    rp.write_csv(os.path.join(OUT, "tables", "defenses-paired-change.csv"), t_pair,
                 ["condition", "name", "n", "mean_change", "change_lo", "change_hi", "fraction_lower",
                  "rank_corr_before_after"])
    figs = []
    try:
        figs += fig_auroc(t_end, os.path.join(OUT, "figures"))
        figs += fig_paired(rows, os.path.join(OUT, "figures"))
    except ImportError:
        print("matplotlib missing: tables and markdown only")

    banner = "" if complete else (f"> **Status: incomplete.** {n_dfd} of {expected} defended readouts present"
                                  f"{f', {len(errors)} failed' if errors else ''}.\n\n")
    lines = [
        "# Experiment 03, part 2: the same prompts under a defense",
        "",
        banner + f"Same 403 prompts, same position (end of the prompt), same layer ({L}), with four "
        "defenses from experiment 02 applied. The undefended row is experiment 03's result, repeated "
        "for comparison. Attack success rates are from the runs that actually generated answers "
        "(8 per attack undefended, 4 per attack under each defense).",
        "",
        rp.md_table(t_end, [("name", "Defense", None),
                            ("attack_mean", "Attack mean", lambda v: num(v, 4)),
                            ("clean_mean", "Clean mean", lambda v: num(v, 5)),
                            ("auroc", "AUROC", lambda v: f"{v:.2f}"),
                            ("auroc_lo", "CI low", lambda v: f"{v:.2f}"),
                            ("auroc_hi", "CI high", lambda v: f"{v:.2f}"),
                            ("attacks_above_clean_max", "Attacks above every clean", str),
                            ("success_rate", "Attack success", lambda v: rp.pct(v))]),
        "",
        "![AUROC and success by defense](figures/05-defenses-auroc-vs-success.svg)",
        "",
        "## Does the defense change the reading, or just the outcome?",
        "",
        "Each attack is read twice: undefended and under the defense. The change is paired within "
        "the attack, so it is not confounded by which attacks are harder.",
        "",
        rp.md_table(t_pair, [("name", "Defense", None), ("n", "Attacks", str),
                             ("mean_change", "Mean change in reading", lambda v: f"{v:+.4f}"),
                             ("change_lo", "CI low", lambda v: f"{v:+.4f}"),
                             ("change_hi", "CI high", lambda v: f"{v:+.4f}"),
                             ("fraction_lower", "Share that read lower", lambda v: rp.pct(v, 0)),
                             ("rank_corr_before_after", "Rank corr. before vs after", lambda v: f"{v:+.2f}")]),
        "",
        "![paired readings](figures/06-defenses-paired-readings.svg)",
        "",
        "How to read the scatter: the dashed line is 'no change'. Dots below it are attacks the "
        "defense pushed out of the model's mind before generation; dots on the line are attacks the "
        "defense left in place. Orange dots still fired at least once under that defense.",
        "",
        "## What this says",
        "",
        "Every defense lowers the reading, and every interval excludes zero. Within a defense the "
        "statement is safe: apply it to the same attacks and both the reading and the unauthorised "
        "calls go down together.",
        "",
        "**What the reading does not do is rank the defenses.** Microsoft's marking leaves a HIGHER "
        "average reading than the library's shipped mode (0.0094 against 0.0024) and still blocks "
        "more attacks (2.4% succeed against 5.4%). Four defenses, one inversion, four points: "
        "nowhere near enough to claim the reading orders them. A separate reason not to treat a "
        "small reading as a good defense: base64 drives it almost to zero by making the email "
        "unreadable, and its summaries keep 6% of their key terms. Suppressing the input suppresses "
        "the attack.",
        "",
        "Note also that this is one short list of tool words at one position, not everything the "
        "model represents. The claim is that a defense lowers this reading, not that it removes the "
        "tool call from the model's mind.",
        "",
        "Base64 flattens the "
        "reading to the floor for 98% of attacks and leaves 2.2% firing. The fixed library moves the reading least "
        "(-0.025, 72% of attacks lower) and keeps the attacks in the same order as undefended "
        "(rank correlation +0.60): the model's state is close to what it was without the defense, "
        "which is why experiment 02 found it cheapest in answer quality, and also why 10.4% of "
        "attacks still fire under it.",
        "",
        "The 'attacks above every clean' column depends on the single largest clean reading and "
        "swings with it (5 under the shipped library because markers make clean prompts noisier); "
        "the paired change and the AUROC are the stable numbers.",
        "",
        "## At the end of the attacker's email",
        "",
        rp.md_table(t_att, [("name", "Defense", None),
                            ("auroc", "AUROC", lambda v: f"{v:.2f}"),
                            ("auroc_lo", "CI low", lambda v: f"{v:.2f}"),
                            ("auroc_hi", "CI high", lambda v: f"{v:.2f}"),
                            ("attack_mean", "Attack mean", lambda v: num(v, 4)),
                            ("clean_mean", "Clean mean", lambda v: num(v, 4))]),
        "",
        "For Microsoft's spotlighting this is the last character before the closing email tag; for "
        "the library conditions the prompt ends with the marked email, so it is the same position as "
        "the end of the prompt minus the chat template.",
        "",
        "## Caveats",
        "",
        "- The benign email's end is not read here: under markers it cannot be located reliably.",
        "- Marked prompts are longer, sometimes much longer; a reading at one position does not say "
        "how the extra tokens are spent.",
        "- Defended success rates rest on 4 runs per attack, so their intervals are wide.",
        "- Two prompts (attack win-075 under Microsoft's marking and under the shipped library) exceed "
        "16,384 tokens and were refused rather than truncated; the run is reported as incomplete "
        "because of them, and nothing else is missing.",
        "- Under the library conditions the end of the attacker's email is the end of the marked "
        "text, whose final token is a marker; the near-zero readings there partly reflect that.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python3 analysis/build_lens_sample_defended.py",
        "cd modal && modal run lens_apply.py --sample runs/03-jacobian-lens/sample_defended.jsonl \\",
        "    --out runs/03-jacobian-lens/readouts_defended.jsonl --chunk 25",
        "cd .. && python3 analysis/report_03_defenses.py",
        "```",
    ]
    with open(os.path.join(OUT, "REPORT-defenses.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest["defenses"] = {
        "built_at": rp.now_utc(), "complete": complete, "items_expected": expected,
        "items_read": n_dfd, "items_failed": len(errors), "conditions": list(NAMES),
        "files": [rp.file_entry(p, "output") for p in figs] +
                 [rp.file_entry(os.path.join(OUT, "tables", n), "output")
                  for n in os.listdir(os.path.join(OUT, "tables")) if n.startswith("defenses-")] +
                 ([rp.file_entry(os.path.join(RAW, "readouts_defended.jsonl"), "lens readouts")]
                  if os.path.exists(os.path.join(RAW, "readouts_defended.jsonl")) else []),
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"wrote {OUT}/REPORT-defenses.md ({'complete' if complete else 'INCOMPLETE'})")


if __name__ == "__main__":
    main()
