"""Generic generation job: run Phi-3 on any prompt file with the challenge's settings.

Every experiment after 01 uses this. (phase1_reproduce.py is kept unchanged as the exact
code that produced experiment 01.)

Input:   a JSONL prompt file; each row needs "id", "prompt", "tool_name"
Output:  a JSONL results file; one row per generated answer, carrying the input row's
         id / base_id / kind / condition plus the answer, parsed tool calls and scores

Generation matches the challenge agent (msref/config.yaml): top_p 0.92, at most 500 new
tokens, sampling on, temperature 1.0 unless overridden. The whole prompt goes in one user
turn, because Phi-3's chat template drops system messages.

Re-running the same command resumes: finished prompts are skipped and new answers are
appended. A dropped connection or a failed batch only costs the prompts in flight.

Run from modal/:
  modal run generate.py --sample runs/02-library-defenses/prompts.jsonl \
      --out runs/02-library-defenses/generations.jsonl --n-samples 4
  add --limit 6 for a smoke test, --condition a,b to restrict conditions
"""
import collections
import json
import os
import time

import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOP_P = 0.92
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 500
TOKEN_BUDGET = 64_000  # (longest prompt in batch + new tokens) x sequences per generate call
CARRY = ("base_id", "kind", "condition", "recorded_label")  # copied from input rows to results

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers>=5.5", "accelerate", "jinja2", "pydantic-core")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("common", "hfload", "llmail_prompt")  # Modal >=1.0 only uploads the entry file
)


@app.cls(image=image, gpu="A100-80GB", volumes={MODEL_DIR: volume}, memory=32768,
         timeout=90 * 60, max_containers=8, scaledown_window=60)
class Generator:
    @modal.enter()
    def load(self):
        import torch
        import transformers
        import hfload
        cfg = hfload.load_config(MODEL_PATH)
        self.tok = hfload.load_tokenizer(MODEL_PATH, cfg)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = hfload.load_model(MODEL_PATH, cfg)
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.runtime = (f"torch {torch.__version__}, transformers {transformers.__version__}, "
                        f"{torch.cuda.get_device_name(0)}, bfloat16")

    def _generate(self, texts, n_ret, temperature):
        import torch
        enc = self.tok(texts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **enc, do_sample=True, top_p=TOP_P, temperature=temperature,
                max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=n_ret,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos)
        return gen[:, enc["input_ids"].shape[1]:].tolist()

    def _generate_batch(self, batch, texts, n_samples, temperature):
        """[(item, sample_index, new_token_ids)]. On CUDA out-of-memory, falls back to one
        prompt at a time with fewer answers per call, halving down to one."""
        import torch
        try:
            new = self._generate([texts[it["id"]] for it in batch], n_samples, temperature)
            return [(batch[j // n_samples], j % n_samples, ids) for j, ids in enumerate(new)]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        out = []
        for it in batch:
            per_call, done = n_samples, 0
            while done < n_samples:
                k = min(per_call, n_samples - done)
                try:
                    new = self._generate([texts[it["id"]]], k, temperature)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if per_call == 1:
                        raise
                    per_call = max(1, per_call // 2)
                    continue
                out.extend((it, done + s, ids) for s, ids in enumerate(new))
                done += k
        return out

    @modal.method()
    def run(self, items, n_samples, seed, temperature=TEMPERATURE):
        import torch
        import llmail_prompt as lp
        torch.manual_seed(seed)
        stop = set(self.eos) | {self.tok.pad_token_id}

        texts = {it["id"]: self.tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}], add_generation_prompt=True, tokenize=False)
            for it in items}
        lens = {k: len(self.tok(t, add_special_tokens=False)["input_ids"]) for k, t in texts.items()}

        batches, cur = [], []
        for it in sorted(items, key=lambda x: lens[x["id"]]):
            if cur and (lens[it["id"]] + MAX_NEW_TOKENS) * n_samples * (len(cur) + 1) > TOKEN_BUDGET:
                batches.append(cur)
                cur = []
            cur.append(it)
        if cur:
            batches.append(cur)

        results = []
        for batch in batches:
            t0 = time.time()
            longest = max(lens[it["id"]] for it in batch)
            try:
                triples = self._generate_batch(batch, texts, n_samples, temperature)
            except Exception as e:  # one bad batch must not sink the chunk; these ids rerun on resume
                torch.cuda.empty_cache()
                results.extend({"id": it["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"} for it in batch)
                print(f"batch FAILED: {len(batch)} prompts, longest {longest} tokens, {type(e).__name__}", flush=True)
                continue
            for it, s, ids in triples:
                n_tok = next((k for k, t in enumerate(ids) if t in stop), len(ids))
                text = self.tok.decode(ids[:n_tok], skip_special_tokens=True)
                calls = lp.parse_tool_calls(text)
                row = {"id": it["id"], "sample": s}
                row.update({k: it[k] for k in CARRY if k in it})
                row.update({
                    "temperature": temperature, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
                    "prompt_tokens": lens[it["id"]], "new_tokens": n_tok,
                    "hit_max_tokens": n_tok >= MAX_NEW_TOKENS,
                    "tool_call_text_present": '{"type": "function"' in text,
                    "tool_calls": calls,
                    **lp.score(calls, it["tool_name"]),
                    "runtime": self.runtime,
                    "response": text,
                })
                results.append(row)
            print(f"batch: {len(batch)} prompts x {n_samples} samples, longest {longest} tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
        return results


@app.local_entrypoint()
def main(sample: str = "", out: str = "", limit: int = 0, n_samples: int = 8, chunk: int = 50,
         seed: int = 0, temperature: float = TEMPERATURE, condition: str = "", kind: str = "",
         overwrite: bool = False):
    if not sample or not out:
        raise SystemExit("pass --sample <prompts.jsonl> and --out <generations.jsonl>, relative to the repo root")
    sample_path, out_path = os.path.join(ROOT, sample), os.path.join(ROOT, out)
    items = [json.loads(line) for line in open(sample_path)]
    if condition:
        keep = set(condition.split(","))
        items = [it for it in items if it.get("condition") in keep]
    if kind:
        items = [it for it in items if it.get("kind") == kind]
    if limit:  # balanced slice across condition x kind x recorded label, for smoke tests
        by = {}
        for it in items:
            by.setdefault((it.get("condition"), it.get("kind"), it.get("recorded_label")), []).append(it)
        per = max(1, limit // len(by))
        items = [x for group in by.values() for x in group[:per]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "w"
    if os.path.exists(out_path) and not overwrite:
        lines = [json.loads(line) for line in open(out_path)]
        counts = collections.Counter(r["id"] for r in lines)
        finished = {i for i, c in counts.items() if c >= n_samples}
        partial = set(counts) - finished
        if partial:  # drop half-written prompts so they rerun cleanly
            with open(out_path, "w") as f:
                for r in lines:
                    if r["id"] in finished:
                        f.write(json.dumps(r) + "\n")
        items = [it for it in items if it["id"] not in finished]
        mode = "a"
        print(f"resuming {out}: {len(finished)} prompts done, {len(partial)} partial dropped, {len(items)} to run")
    if not items:
        print("nothing left to run")
        return

    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    print(f"{len(items)} prompts x {n_samples} samples at temperature {temperature}, {len(chunks)} chunks -> {out}")
    t0 = time.time()
    failed = set()
    calls = [(c, n_samples, seed + i, temperature) for i, c in enumerate(chunks)]
    with open(out_path, mode) as f:
        for n_done, res in enumerate(Generator().run.starmap(calls, order_outputs=False, return_exceptions=True), 1):
            if isinstance(res, BaseException):
                print(f"chunk {n_done}/{len(chunks)} FAILED remotely: {type(res).__name__}: {str(res)[:200]}")
                continue
            good = [r for r in res if "error" not in r]
            failed |= {r["id"] for r in res if "error" in r}
            for r in good:
                f.write(json.dumps(r) + "\n")
            f.flush()
            sent = sum(r["exfil.sent"] for r in good)
            print(f"chunk {n_done}/{len(chunks)} written: {len(good)} samples, {sent} tool calls ({time.time() - t0:.0f}s)")
    if failed:
        print(f"{len(failed)} prompts failed and were not written; rerun the same command to retry them")
    print("done.")
