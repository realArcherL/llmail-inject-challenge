"""PHASE 1: re-run LLMail-Inject level 1 on Phi-3 with Microsoft's exact prompts.

Input:   ../runs/phase1/sample.jsonl      (from analysis/build_phase1_sample.py)
Output:  ../runs/phase1/results_<tag>.jsonl   one line per generated sample

Generation matches the challenge agent (msref/config.yaml): top_p 0.92, at most
500 new tokens, sampling on. The paper says 1,000 tokens; the released agent code
says 500, and the code is what ran. The agent never set temperature, so Azure's
endpoint default applied; that default is not documented, so we use 1.0.

Run:
  modal run phase1_reproduce.py --limit 4 --n-samples 2 --tag smoke   # a few cents
  modal run phase1_reproduce.py                                       # full run
"""
import json
import os
import time

import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "runs", "phase1", "sample.jsonl")
OUT_DIR = os.path.join(ROOT, "runs", "phase1")

TOP_P = 0.92
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 500
TOKEN_BUDGET = 96_000  # (longest prompt in batch + new tokens) x sequences, keeps KV cache ~20 GB

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers>=5.5", "accelerate", "jinja2", "pydantic-core")
    .add_local_python_source("common", "hfload", "llmail_prompt")  # Modal >=1.0 only uploads the entry file
)


@app.cls(image=image, gpu="A100-80GB", volumes={MODEL_DIR: volume}, memory=32768,
         timeout=90 * 60, max_containers=8, scaledown_window=60)
class Phase1:
    @modal.enter()
    def load(self):
        import hfload
        cfg = hfload.load_config(MODEL_PATH)
        self.tok = hfload.load_tokenizer(MODEL_PATH, cfg)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = hfload.load_model(MODEL_PATH, cfg)
        eos = self.model.generation_config.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]

    @modal.method()
    def run(self, items, n_samples, seed):
        import torch
        import llmail_prompt as lp
        torch.manual_seed(seed)
        stop = set(self.eos) | {self.tok.pad_token_id}

        # Phi-3 has one user turn; the whole challenge prompt lives inside it.
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
            enc = self.tok([texts[it["id"]] for it in batch], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, do_sample=True, top_p=TOP_P, temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=n_samples,
                    pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos)
            new = gen[:, enc["input_ids"].shape[1]:].tolist()
            for j, ids in enumerate(new):
                it = batch[j // n_samples]
                n_tok = next((k for k, t in enumerate(ids) if t in stop), len(ids))
                text = self.tok.decode(ids[:n_tok], skip_special_tokens=True)
                calls = lp.parse_tool_calls(text)
                results.append({
                    "id": it["id"], "sample": j % n_samples,
                    "prompt_tokens": lens[it["id"]], "new_tokens": n_tok,
                    "hit_max_tokens": n_tok >= MAX_NEW_TOKENS,
                    "tool_call_text_present": '{"type": "function"' in text,
                    "tool_calls": calls,
                    **lp.score(calls, it["tool_name"]),
                    "response": text,
                })
            print(f"batch: {len(batch)} prompts x {n_samples} samples, longest {max(lens[i['id']] for i in batch)} tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
        return results


@app.local_entrypoint()
def main(limit: int = 0, n_samples: int = 8, chunk: int = 50, seed: int = 0, tag: str = "full"):
    items = [json.loads(line) for line in open(SAMPLE)]
    if limit:  # balanced slice across condition x recorded label, for smoke tests
        by = {}
        for it in items:
            by.setdefault((it["condition"], it["recorded_label"]), []).append(it)
        per = max(1, limit // len(by))
        items = [x for group in by.values() for x in group[:per]]
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    out_path = os.path.join(OUT_DIR, f"results_{tag}.jsonl")
    print(f"{len(items)} prompts x {n_samples} samples, {len(chunks)} chunks -> {out_path}")

    t0 = time.time()
    with open(out_path, "w") as f:
        for done, res in enumerate(Phase1().run.starmap(
                [(c, n_samples, seed + i) for i, c in enumerate(chunks)], order_outputs=False), 1):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            sent = sum(r["exfil.sent"] for r in res)
            print(f"chunk {done}/{len(chunks)} written: {len(res)} samples, {sent} tool calls ({time.time() - t0:.0f}s)")
    print(f"done. next: python3 analysis/phase1_report.py --tag {tag}")
