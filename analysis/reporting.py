"""Shared helpers for the publishable results/ folders.

One consistent format for every experiment: checksums and a manifest for provenance,
payload-level bootstrap confidence intervals, CSV tables, and figures in one visual style
(light and dark variants, palette checked for colour-vision deficiency and contrast).
"""
import csv
import datetime
import hashlib
import json
import os
import random
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Phi-3 weights are byte-identical on Hugging Face since the first upload (checked 2026-09-11).
MODEL = {
    "id": "microsoft/Phi-3-medium-128k-instruct",
    "weights_uploaded": "2024-05-02",
    "weights_unchanged_since_upload": True,
    "weight_file_lfs_oid_prefixes": {
        "model-00001-of-00006.safetensors": "f438f5fea1b8",
        "model-00002-of-00006.safetensors": "218be0ac9171",
        "model-00003-of-00006.safetensors": "ca6f6b1d156c",
        "model-00004-of-00006.safetensors": "cee66c06fdcb",
        "model-00005-of-00006.safetensors": "e147b780080e",
        "model-00006-of-00006.safetensors": "a97b1288d949",
    },
    "dtype": "bfloat16",
    "loader_note": "config.json RoPE key copied into rope_scaling in memory for transformers>=5.5 (modal/hfload.py)",
}
PROMPT_RECIPE = {
    "source": "microsoft/llmail-inject-challenge",
    "commit": "ad115315c1cb34381d20875d6675a6cfe6ca80fa",
    "vendored_at": "modal/msref/",
    "port": "modal/llmail_prompt.py",
    "layout": "system rules, tool prompt, query and emails in ONE user turn (Phi-3 has no system role)",
}
CHALLENGE_GENERATION = {"top_p": 0.92, "max_new_tokens": 500, "sampling": True,
                        "temperature": "1.0 (agent never set it; Azure's default is undocumented)"}

# Validated with the dataviz palette checker (scripts/validate_palette.js), light and dark.
STYLE = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6", "accent": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5", "accent": "#d95926"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_entry(path, role):
    rows = None
    if path.endswith((".jsonl", ".csv")):
        with open(path, "rb") as f:
            rows = sum(1 for _ in f) - (1 if path.endswith(".csv") else 0)
    return {"path": os.path.relpath(path, ROOT), "role": role, "rows": rows,
            "bytes": os.path.getsize(path), "sha256": sha256(path)}


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap_mean(values, n=2000, seed=0):
    """Mean and 95% percentile interval, resampling payloads (the unit that varies)."""
    values = [v for v in values if v == v]
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng, k = random.Random(seed), len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(values) / k, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def md_table(rows, cols):
    """rows: list of dicts; cols: list of (key, header, formatter)."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(r[k]) if fmt else str(r[k]) for k, _, fmt in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def pct(x, digits=1):
    return "n/a" if x != x else f"{100 * x:.{digits}f}%"


def pct_ci(m, lo, hi):
    return "n/a" if m != m else f"{100*m:.1f}% ({100*lo:.1f} to {100*hi:.1f})"


def figure(dark=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = STYLE["dark" if dark else "light"]
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
        "axes.edgecolor": s["axis"], "axes.labelcolor": s["ink2"], "axes.facecolor": s["surface"],
        "figure.facecolor": s["surface"], "xtick.color": s["muted"], "ytick.color": s["ink"],
        "axes.grid": True, "grid.color": s["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
    })
    return plt, s


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    return paths
