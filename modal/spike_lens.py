"""PHASE 0 SPIKE. One question: does anthropics/jacobian-lens run on Phi-3?

Run:       modal run spike_lens.py
Fallback:  modal run spike_lens.py --model-path Qwen/Qwen2.5-1.5B-Instruct

This is a plumbing test, not a usable lens: 2 prompts, 64 tokens, 3 layers.
Fit cost is one forward plus d_model/dim_batch backward passes per prompt,
so a real Phi-3 lens (d_model 5120) is a Phase 2 job measured in hours.
"""
import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

LENS_REPO = "https://github.com/anthropics/jacobian-lens.git"
SPIKE_LAYERS = [10, 20, 30]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch", "transformers>=5.5", "accelerate", "numpy", "jinja2")
    .run_commands(
        f"git clone --depth 1 {LENS_REPO} /opt/jlens",
        "cd /opt/jlens && pip install -e .",
    )
    .add_local_python_source("common", "hfload")  # Modal >=1.0 only uploads the entry file
)

# jlens skips the first 16 positions (attention sink), so prompts must be well
# over 16 tokens. These are ~60-80 tokens each; max_seq_len caps them at seq_len.
PROMPTS = [
    "Hi team, following up on yesterday's planning meeting. The quarterly budget review "
    "has moved to Thursday afternoon, and finance has asked each group to send its revised "
    "headcount numbers before Wednesday noon. Please also attach the vendor contracts that "
    "are up for renewal so we can compare costs against last year's figures in one place.",
    "The history of the printing press shows how a single technical change can reshape an "
    "entire society. Before movable type, books were copied by hand, which made them rare "
    "and expensive. Within a few decades of Gutenberg's work, printed pamphlets were spreading "
    "news, arguments, and scientific ideas across Europe faster than any authority could control.",
    "Reminder: the office will be closed on Monday for maintenance on the building's electrical "
    "systems. Remote access to the file servers will remain available, but the conference rooms "
    "and the kitchen will be locked. If you need to collect equipment, please do so before "
    "Friday evening and let facilities know which items you are taking home with you.",
]


@app.function(image=image, gpu="A100-80GB", volumes={MODEL_DIR: volume},
              memory=32768, timeout=60 * 60)
def spike(model_path: str = MODEL_PATH, n_prompts: int = 2, seq_len: int = 64):
    """Returns {step: (status, text)}. Only plain strings travel back to your Mac,
    which has no torch or jlens installed to unpack anything richer."""
    import time, traceback
    results = {}

    def step(name, fn):
        t0 = time.time()
        try:
            out = fn()
            summary = str(out)[:200]
            results[name] = ("ok", summary)
            print(f"[ok]   {name} ({time.time() - t0:.0f}s): {summary}")
            return out
        except Exception as e:
            results[name] = ("FAIL", f"{type(e).__name__}: {e}"[:500])
            print(f"[FAIL] {name} ({time.time() - t0:.0f}s): {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    jlens = step("import jlens", lambda: __import__("jlens"))
    hfload = step("import hfload", lambda: __import__("hfload"))
    if jlens is None or hfload is None:
        return results

    cfg = step("load config (patched)", lambda: hfload.load_config(model_path))
    if cfg is None:
        return results
    tok = step("load tokenizer", lambda: hfload.load_tokenizer(model_path, cfg))
    hf = step("load model", lambda: hfload.load_model(model_path, cfg))
    if tok is None or hf is None:
        return results
    step("model shape", lambda: f"{cfg.model_type}, {cfg.num_hidden_layers} layers, d={cfg.hidden_size}")

    model = step("jlens.from_hf", lambda: jlens.from_hf(hf, tok))
    if model is None:
        print("\n>>> jlens could not wrap this architecture. Try the Qwen fallback. <<<")
        return results

    layers = [l for l in SPIKE_LAYERS if l < cfg.num_hidden_layers - 1]
    lens = step("jlens.fit (tiny)", lambda: jlens.fit(
        model, prompts=PROMPTS[:n_prompts], source_layers=layers,
        max_seq_len=seq_len, dim_batch=16, checkpoint_every=None))
    if lens is None:
        return results

    def apply():
        lens_logits, _, _ = lens.apply(model, "Fact: the currency used in Japan is", positions=[-2])
        readout = {}
        for layer, lg in sorted(lens_logits.items()):
            row = lg.reshape(-1, lg.shape[-1])[0]  # first requested position, whatever the rank
            readout[int(layer)] = [tok.decode([t]) for t in row.topk(3).indices.tolist()]
        return readout

    readout = step("lens.apply", apply)
    if readout:
        print("\nlayer -> top 3 words that activation is poised to make the model say")
        for layer, words in readout.items():
            print(f"  L{layer:<3} {words}")
    return results


@app.local_entrypoint()
def main(model_path: str = MODEL_PATH, n_prompts: int = 2, seq_len: int = 64):
    r = spike.remote(model_path, n_prompts, seq_len)
    bad = [k for k, (s, _) in r.items() if s == "FAIL"]
    print("\n" + "=" * 60)
    if bad:
        print(f"RESULT: FAIL at step -> {bad[0]}")
        print(f"        {r[bad[0]][1][:300]}")
    else:
        print(f"RESULT: PASS. The Jacobian lens runs on {model_path}.")
    print("=" * 60)
