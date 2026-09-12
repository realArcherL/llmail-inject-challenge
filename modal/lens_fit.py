"""Experiment 03, step 1: fit a Jacobian lens for Phi-3, then check it still reads at depth.

The lens (github.com/anthropics/jacobian-lens) carries an early-layer residual into the final-layer
basis using the average input-output Jacobian, so the model's own unembedding can decode it. It shows
what an activation is poised to make the model say, now or later.

Cost is one forward pass plus ceil(d_model / dim_batch) backward passes per fitting prompt, so the
work is split across containers and merged with JacobianLens.merge.

The lens is written to the phi3-weights volume at /models/lens/<tag>.pt, where the readout job finds it.

Run:
  modal run lens_fit.py                      # 100 wikitext passages, 4 shards, about 45 min, ~$7
  modal run lens_fit.py --probe-only         # re-check an existing lens, pennies
"""
import json
import os
import time

import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

LENS_DIR = f"{MODEL_DIR}/lens"
LENS_REPO = "https://github.com/anthropics/jacobian-lens.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch", "transformers>=5.5", "accelerate", "jinja2", "numpy", "datasets")
    .run_commands(
        f"git clone --depth 1 {LENS_REPO} /opt/jlens",
        "cd /opt/jlens && pip install -e .",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("common", "hfload")
)

# Known-answer probes. The lens is fitted on short passages but read far deeper than that, so the
# same question is asked at the start of a prompt and again after ~1,200 tokens of filler.
PROBES = [
    ("capital", "Fact: the capital city of France is", " Paris"),
    ("currency", "Fact: the currency used in Japan is the", " yen"),
]
FILLER = (
    "The river system supports a mix of farms, small towns and light industry along its banks. "
    "Rainfall arrives mostly in spring, and the water level falls through the dry months that follow. "
    "Local councils have argued about the bridge for years without reaching a decision. "
)


@app.cls(image=image, gpu="A100-80GB", volumes={MODEL_DIR: volume}, memory=65536,
         timeout=150 * 60, max_containers=8, scaledown_window=60)
class Lens:
    @modal.enter()
    def load(self):
        import jlens
        import hfload
        cfg = hfload.load_config(MODEL_PATH)
        self.tok = hfload.load_tokenizer(MODEL_PATH, cfg)
        self.hf = hfload.load_model(MODEL_PATH, cfg)
        self.model = jlens.from_hf(self.hf, self.tok)
        self.jlens = jlens
        os.makedirs(LENS_DIR, exist_ok=True)

    @modal.method()
    def fit_shard(self, shard, shards, n_prompts, max_seq_len, dim_batch, tag):
        """Fit on this shard's slice of the corpus and save it to the volume."""
        from jlens.examples import load_wikitext_prompts
        prompts = load_wikitext_prompts(n_prompts=n_prompts)[shard::shards]
        t0 = time.time()
        lens = self.jlens.fit(
            self.model, prompts, max_seq_len=max_seq_len, dim_batch=dim_batch,
            checkpoint_path=f"{LENS_DIR}/{tag}.shard{shard}.ckpt", checkpoint_every=5)
        path = f"{LENS_DIR}/{tag}.shard{shard}.pt"
        lens.save(path)
        volume.commit()
        took = time.time() - t0
        print(f"shard {shard}: {len(prompts)} prompts in {took/60:.1f} min -> {path}", flush=True)
        return {"shard": shard, "prompts": len(prompts), "minutes": round(took / 60, 1), "path": path}

    @modal.method()
    def merge_and_probe(self, tag, shards, layers=None):
        """Merge the shards into one lens, save it, then read the known-answer probes."""
        volume.reload()
        paths = [f"{LENS_DIR}/{tag}.shard{s}.pt" for s in range(shards)]
        lenses = [self.jlens.JacobianLens.load(p) for p in paths]
        lens = lenses[0] if len(lenses) == 1 else self.jlens.JacobianLens.merge(lenses)
        out_path = f"{LENS_DIR}/{tag}.pt"
        lens.save(out_path)
        volume.commit()
        print(f"merged {len(lenses)} shards -> {out_path}", flush=True)
        return {"lens": out_path, "probes": self._probe(lens, layers)}

    @modal.method()
    def probe(self, tag, layers=None):
        volume.reload()
        lens = self.jlens.JacobianLens.load(f"{LENS_DIR}/{tag}.pt")
        return {"lens": f"{LENS_DIR}/{tag}.pt", "probes": self._probe(lens, layers)}

    def _probe(self, lens, layers=None):
        """Top words the lens reads at the last position, shallow and deep, per layer."""
        results = {}
        for name, question, answer in PROBES:
            # shallow, then about the length of our real prompts, then well past it
            for depth, text in (("shallow", question),
                                ("mid_1200_tokens", FILLER * 20 + " " + question),
                                ("deep_3700_tokens", FILLER * 60 + " " + question)):
                lens_logits, model_logits, ids = lens.apply(
                    self.model, text, layers=layers, positions=[-1], max_seq_len=4096)
                reading = {}
                for layer, lg in sorted(lens_logits.items()):
                    v = lg.reshape(-1, lg.shape[-1])[0].float()
                    reading[int(layer)] = [self.tok.decode([i]) for i in v.topk(5).indices.tolist()]
                final = model_logits.reshape(-1, model_logits.shape[-1])[0].float()
                results[f"{name}/{depth}"] = {
                    "tokens_in_prompt": int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids),
                    "model_says": [self.tok.decode([i]) for i in final.topk(5).indices.tolist()],
                    "expected": answer,
                    "lens_by_layer": reading,
                }
        return results


@app.local_entrypoint()
def main(n_prompts: int = 100, shards: int = 4, max_seq_len: int = 128, dim_batch: int = 32,
         tag: str = "phi3-wikitext100", probe_only: bool = False, layers: str = "8,16,24,32,38"):  # 39 is the read-out target, so 38 is the deepest source layer
    picked = [int(x) for x in layers.split(",") if x.strip()]
    lens = Lens()
    if probe_only:
        print(json.dumps(lens.probe.remote(tag, picked), indent=2)[:4000])
        return

    print(f"fitting '{tag}': {n_prompts} passages of {max_seq_len} tokens across {shards} shards")
    t0 = time.time()
    for r in lens.fit_shard.starmap(
            [(s, shards, n_prompts, max_seq_len, dim_batch, tag) for s in range(shards)],
            order_outputs=False):
        print(f"  shard {r['shard']} done: {r['prompts']} prompts, {r['minutes']} min")
    print(f"all shards fitted in {(time.time() - t0) / 60:.1f} min; merging")
    out = lens.merge_and_probe.remote(tag, shards, picked)
    print(json.dumps(out, indent=2)[:6000])
    print("\nread the probes above: the deep readout should name the same answer as the shallow one.")
