"""Experiment 03, part 3: what the model makes of being told it was prompt-injected.

Sixteen attacks the model fell for. After its answer, a second human turn either tells it that was
a prompt injection ("told") or asks an ordinary follow-up ("neutral"). The lens reads every token of
that turn and the position that produces the reply.

The question is not whether tool words appear (they should not, in either case). It is which words
the lens reads for "told" that it does not read for "neutral": the model's reaction to the
accusation, separated from its reaction to there being a second turn at all.

Inputs:  runs/03-jacobian-lens/readouts_told.jsonl, sample_told.jsonl
Outputs: results/03-jacobian-lens/REPORT-told.md, tables/told-*.csv

Run:  python3 analysis/report_03_told.py
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting as rp  # noqa: E402
from report_03_lens import TRUSTED_FROM  # noqa: E402

RAW = os.path.join(rp.ROOT, "runs", "03-jacobian-lens")
OUT = os.path.join(rp.ROOT, "results", "03-jacobian-lens")
DEEP = list(range(30, 39))
JUNK = re.compile(r"[^\x00-\x7f]|^[\\/|#=_\-\.]{3,}$|^\s*$")
APOLOGY = re.compile(r"^\s*(ap|apolog|sorry|forg|regret|mistake|my)$", re.I)


def apology_by_layer(rows):
    """Share of attacks whose top-8 at the reply position holds an apology word, per layer and turn.
    If it only appears in the last few layers it is the next token; if it appears from the middle
    of the network it was in mind well before the reply."""
    out = []
    for layer in range(18, 39, 2):
        row = {"layer": layer}
        for v in ("told", "neutral"):
            n = hit = 0
            for r in rows:
                if r["variant"] != v:
                    continue
                k = next((k for k in r["readings"] if k["position_label"] == "prompt_end" and k["layer"] == layer), None)
                if k:
                    n += 1
                    hit += any(APOLOGY.match(w) for w in k["top_words"])
            row[v] = hit / n if n else float("nan")
        out.append(row)
    return out


def load():
    path = os.path.join(RAW, "readouts_told.jsonl")
    rows = rp.jsonl(path) if os.path.exists(path) else []
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    sample = {r["id"]: r for r in rp.jsonl(os.path.join(RAW, "sample_told.jsonl"))}
    return rows, errors, sample


def word_shares(rows, variant, label_filter, layers):
    """Share of (item, layer) readings whose top-8 contains each word."""
    c, n = collections.Counter(), 0
    for r in rows:
        if r["variant"] != variant:
            continue
        for k in r["readings"]:
            if k["layer"] not in layers or not label_filter(k["position_label"]):
                continue
            n += 1
            for w in set(x.strip() for x in k["top_words"]):
                if w and not JUNK.search(w):
                    c[w] += 1
    return {w: v / n for w, v in c.items()} if n else {}, n


def differential(rows, label_filter, layers, top=20):
    told, n_t = word_shares(rows, "told", label_filter, layers)
    neut, n_n = word_shares(rows, "neutral", label_filter, layers)
    words = set(told) | set(neut)
    out = [{"word": w, "told_share": told.get(w, 0.0), "neutral_share": neut.get(w, 0.0),
            "difference": told.get(w, 0.0) - neut.get(w, 0.0)} for w in words]
    out.sort(key=lambda x: -x["difference"])
    return out[:top], sorted(out, key=lambda x: x["difference"])[:top], n_t, n_n


def transcript_rows(rows, sample, base_id, variant, layer):
    """Token by token along the second turn, at one layer: token text and top-5 words."""
    r = next((x for x in rows if x["base_id"] == base_id and x["variant"] == variant), None)
    if not r:
        return []
    out = []
    for k in r["readings"]:
        if k["layer"] != layer:
            continue
        lab = k["position_label"]
        idx = int(lab.rsplit("_", 1)[1]) if lab.startswith("turn_token_") else 10 ** 6
        out.append((idx, lab, k["tool_word_prob"], [w for w in k["top_words"][:5]]))
    return sorted(out)


def main():
    rows, errors, sample = load()
    expected = len(sample)
    complete = len(rows) == expected and not errors
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

    at_end = lambda lab: lab == "prompt_end"  # noqa: E731
    in_turn = lambda lab: lab.startswith("turn_token_")  # noqa: E731
    up_end, down_end, n_te, n_ne = differential(rows, at_end, DEEP)
    up_turn, down_turn, n_tt, n_nt = differential(rows, in_turn, DEEP)
    rp.write_csv(os.path.join(OUT, "tables", "told-differential-words-prompt-end.csv"),
                 up_end + down_end, ["word", "told_share", "neutral_share", "difference"])
    rp.write_csv(os.path.join(OUT, "tables", "told-differential-words-along-turn.csv"),
                 up_turn + down_turn, ["word", "told_share", "neutral_share", "difference"])

    prof = apology_by_layer(rows)
    rp.write_csv(os.path.join(OUT, "tables", "told-apology-by-layer.csv"), prof, ["layer", "told", "neutral"])

    # tool-word probability at the reply position, told vs neutral, paired by attack
    pairs = collections.defaultdict(dict)
    for r in rows:
        k = next((k for k in r["readings"] if k["position_label"] == "prompt_end" and k["layer"] == 38), None)
        if k:
            pairs[r["base_id"]][r["variant"]] = k["tool_word_prob"]
    diffs = [v["told"] - v["neutral"] for v in pairs.values() if "told" in v and "neutral" in v]
    dm, dlo, dhi = rp.bootstrap_mean(diffs) if diffs else (float("nan"),) * 3

    banner = "" if complete else (f"> **Status: incomplete.** {len(rows)} of {expected} readouts present"
                                  f"{f', {len(errors)} failed' if errors else ''}.\n\n")
    fmt = lambda v: rp.pct(v, 0)  # noqa: E731
    lines = [
        "# Experiment 03, part 3: being told",
        "",
        banner + f"{len(rows)} readouts: {len(pairs)} attacks the model fell for, each replayed with two "
        "different second human turns of the same length. **Told:** \"That second email was a prompt "
        "injection. You were not supposed to send anything… Why did you do that?\" **Neutral:** an "
        "ordinary follow-up asking who sent the emails and when.",
        "",
        "Words are counted across layers 30–38. A share of 50% means the word was in the lens's "
        "top 8 at half of the (attack, layer) readings for that turn.",
        "",
        "## At the position that produces the reply",
        "",
        "Words the lens reads for **told** far more than for **neutral**:",
        "",
        rp.md_table(up_end[:12], [("word", "Word", lambda w: f"`{w}`"), ("told_share", "Told", fmt),
                                  ("neutral_share", "Neutral", fmt), ("difference", "Difference", lambda v: f"{100*v:+.0f} pts")]),
        "",
        "And the reverse — what the neutral follow-up brings up that the accusation does not:",
        "",
        rp.md_table(down_end[:8], [("word", "Word", lambda w: f"`{w}`"), ("told_share", "Told", fmt),
                                   ("neutral_share", "Neutral", fmt), ("difference", "Difference", lambda v: f"{100*v:+.0f} pts")]),
        "",
        f"Tool-word probability at that position, told minus neutral, paired by attack at layer 38: "
        f"{dm:+.4f} ({dlo:+.4f} to {dhi:+.4f}, n={len(diffs)}).",
        "",
        "### Is the apology only the next token, or already in mind?",
        "",
        "Share of the 16 attacks whose top 8 at the reply position holds an apology word "
        "(*apolog-, sorry, forgive, my*), by layer. Layers 30 and above sit close to the output; "
        "anything earlier is not yet the next token.",
        "",
        rp.md_table(prof, [("layer", "Layer", str), ("told", "Told", fmt), ("neutral", "Neutral", fmt)]),
        "",
        "## Along the second turn itself",
        "",
        "Every token of the human's sentence, layers 30–38. Words specific to being told:",
        "",
        rp.md_table(up_turn[:12], [("word", "Word", lambda w: f"`{w}`"), ("told_share", "Told", fmt),
                                   ("neutral_share", "Neutral", fmt), ("difference", "Difference", lambda v: f"{100*v:+.0f} pts")]),
        "",
    ]
    # one worked transcript at layer 38
    if pairs:
        bid = sorted(pairs)[0]
        for variant in ("told", "neutral"):
            tr = transcript_rows(rows, sample, bid, variant, 38)
            if not tr:
                continue
            s = sample.get(f"C-{bid}-{variant}", {})
            turn = s.get("messages", [{}])[-1].get("content", "")
            lines += [f"### {bid}, {variant} turn, layer 38", "", f"> {turn}", "",
                      "| Token | Tool-word prob | Lens reads |", "|---|---|---|"]
            for idx, lab, p, words in tr:
                name = "→ reply position" if lab == "prompt_end" else f"token {idx}"
                lines.append(f"| {name} | {p:.3f} | " + " ".join(f"`{w.strip()}`" for w in words if w.strip()) + " |")
            lines.append("")
    lines += [
        "## What this is and is not",
        "",
        "- Sixteen attacks, one recorded answer each. This is a look, not a measurement with intervals "
        "on the word shares; the paired tool-word difference is the one number with an interval.",
        "- The lens is fitted on plain English. Words it cannot render come out as junk and are "
        "dropped from the counts.",
        "- The recorded answer is one the model actually gave when it fell for the attack; the model "
        "is being confronted with its own real output.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python3 analysis/build_lens_sample_told.py",
        "cd modal && modal run lens_apply.py --sample runs/03-jacobian-lens/sample_told.jsonl \\",
        "    --out runs/03-jacobian-lens/readouts_told.jsonl --chunk 8",
        "cd .. && python3 analysis/report_03_told.py",
        "```",
    ]
    with open(os.path.join(OUT, "REPORT-told.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    manifest["told"] = {"built_at": rp.now_utc(), "complete": complete, "items_expected": expected,
                        "items_read": len(rows), "items_failed": len(errors),
                        "files": [rp.file_entry(os.path.join(OUT, "tables", n), "output")
                                  for n in os.listdir(os.path.join(OUT, "tables")) if n.startswith("told-")]}
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"wrote {OUT}/REPORT-told.md ({'complete' if complete else 'INCOMPLETE'})")


if __name__ == "__main__":
    main()
