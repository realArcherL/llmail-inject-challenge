"""Pick which experiment 02 prompts actually need generating.

A lib_sanitize prompt that is byte-identical to its no-defense prompt means the library
found nothing to strip. Identical input means an identical answer distribution, so it is
not re-generated; the report reuses the no-defense answers for it and says so.

Writes runs/02-library-defenses/prompts_run.jsonl.
Run:  python3 analysis/make_run_list.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "runs", "02-library-defenses")

parts = {}
for line in open(os.path.join(RAW, "parts.jsonl")):
    r = json.loads(line)
    parts[r["base_id"]] = r["prompt_none"]

keep, skipped = [], 0
for line in open(os.path.join(RAW, "prompts.jsonl")):
    r = json.loads(line)
    if r["condition"] == "lib_sanitize" and r["prompt"] == parts[r["base_id"]]:
        skipped += 1
        continue
    keep.append(line)
with open(os.path.join(RAW, "prompts_run.jsonl"), "w") as f:
    f.writelines(keep)
print(f"prompts_run.jsonl: {len(keep)} prompts to generate; {skipped} sanitize-only prompts identical to no-defense, reused instead")
