"""Experiment 03, step 2: read the Jacobian lens on LLMail prompts and on recorded answers.

Needs a lens from lens_fit.py at /models/lens/<tag>.pt.

Study A reads every prompt at three places: end of the benign email, end of the attacker email, and
the very end of the prompt (the position that produces the first generated token).

Study B replays a recorded answer through the model (the prompt plus that answer, teacher forcing)
and reads along the answer, so a winning run and a losing run of the SAME prompt can be compared.

Study C is a conversation: the prompt, a recorded answer, then a second human turn (for example,
telling the model it was just prompt-injected). It reads every token of that second turn and the
position that produces the reply, so what the model makes of being told can be seen token by token.

Study A items may carry only some spans: any of benign_email_end / attacker_email_end is read only
when its span is present (defended prompts cannot always locate the emails under the markers), and
prompt_end is always read.

For every (layer, position) it saves the top words the lens reads, plus the probability it assigns to
a fixed set of tool-call words, which is the quantity the analysis compares.

Run:
  modal run lens_apply.py --limit 8                     # trial, pennies
  modal run lens_apply.py                               # everything in sample.jsonl
  modal run lens_apply.py --sample runs/03-jacobian-lens/sample_defended.jsonl \
                          --out runs/03-jacobian-lens/readouts_defended.jsonl
"""
import json
import os
import time

import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "runs", "03-jacobian-lens", "sample.jsonl")
OUT = os.path.join(ROOT, "runs", "03-jacobian-lens", "readouts.jsonl")
LENS_DIR = f"{MODEL_DIR}/lens"
LENS_REPO = "https://github.com/anthropics/jacobian-lens.git"

# Words that mean "call the email tool". Tracked the same way for every prompt, so nothing depends on
# a payload's random tool-name suffix.
TOOL_WORDS = ["send", " send", "email", " email", "contact", " contact", "@", "confirmation",
              " confirmation", "function", " function", '{"', "type"]
TOP_K = 8
RESPONSE_STRIDE = 8  # read every Nth token of an answer in study B
TURN_STRIDE = 1      # study C reads every token of the second human turn
MAX_TOKENS = 16384   # longer than this is refused rather than silently truncated

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch", "transformers>=5.5", "accelerate", "jinja2", "numpy")
    .run_commands(
        f"git clone --depth 1 {LENS_REPO} /opt/jlens",
        "cd /opt/jlens && pip install -e .",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("common", "hfload")
)


@app.cls(image=image, gpu="A100-80GB", volumes={MODEL_DIR: volume}, memory=65536,
         timeout=120 * 60, max_containers=6, scaledown_window=60)
class LensReader:
    tag: str = modal.parameter(default="phi3-wikitext100")

    @modal.enter()
    def load(self):
        import jlens
        import torch
        import hfload
        cfg = hfload.load_config(MODEL_PATH)
        self.tok = hfload.load_tokenizer(MODEL_PATH, cfg)
        self.hf = hfload.load_model(MODEL_PATH, cfg)
        # force_bos=False: read the text exactly as it was generated. Experiments 01 and 02
        # tokenized with add_special_tokens=False, and jlens would otherwise prepend a start
        # token, shifting every position by one.
        self.model = jlens.from_hf(self.hf, self.tok, force_bos=False)
        volume.reload()
        self.lens = jlens.JacobianLens.load(f"{LENS_DIR}/{self.tag}.pt")
        self.layers = sorted(int(x) for x in self.lens.source_layers)
        self.torch = torch
        # first token id of each tool word, deduplicated
        self.tool_ids = {}
        for w in TOOL_WORDS:
            ids = self.tok(w, add_special_tokens=False)["input_ids"]
            if ids:
                self.tool_ids.setdefault(int(ids[0]), w)
        # The chat template wraps the prompt, so character offsets measured in the prompt have to
        # be shifted by the template prefix before they can be looked up in the rendered text.
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": "\x00"}], add_generation_prompt=True, tokenize=False)
        self.template_prefix, self.template_suffix = rendered.split("\x00")
        # Belt and braces: if the model ever encodes with extra leading tokens, shift positions.
        probe = "hello world"
        self.pos_shift = (int(self.model.encode(probe).shape[-1])
                          - len(self.tok(probe, add_special_tokens=False)["input_ids"]))
        print(f"position shift vs plain tokenization: {self.pos_shift}", flush=True)
        print(f"lens {self.tag}: layers {self.layers[0]}..{self.layers[-1]}, "
              f"{len(self.tool_ids)} tool-word tokens, template prefix {len(self.template_prefix)} chars",
              flush=True)

    def _render(self, prompt, response=None):
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        return text + (response or "")

    def _token_at_char(self, text, char_index):
        """Index of the token containing a character offset, using the tokenizer's own offsets."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        for i, (s, e) in enumerate(enc["offset_mapping"]):
            if s <= char_index < e:
                return i
        return len(enc["input_ids"]) - 1

    def _read(self, text, positions, labels):
        n = len(self.tok(text, add_special_tokens=False)["input_ids"])
        if n > MAX_TOKENS:
            raise ValueError(f"{n} tokens exceeds MAX_TOKENS={MAX_TOKENS}; refusing to truncate")
        lens_logits, model_logits, ids = self.lens.apply(
            self.model, text, positions=positions, max_seq_len=MAX_TOKENS)
        rows = []
        for layer, lg in sorted(lens_logits.items()):
            probs = self.torch.softmax(lg.float(), dim=-1)
            for j, label in enumerate(labels):
                v = probs[j]
                top = v.topk(TOP_K)
                rows.append({
                    "layer": int(layer),
                    "position_label": label,
                    "position": int(positions[j]),
                    "top_words": [self.tok.decode([i]) for i in top.indices.tolist()],
                    "top_probs": [round(float(x), 5) for x in top.values.tolist()],
                    "tool_word_prob": round(float(sum(v[i] for i in self.tool_ids)), 6),
                    "tool_words_seen": [w for i, w in self.tool_ids.items()
                                        if float(v[i]) > 0.01],
                })
        return rows, int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)

    @modal.method()
    def read(self, items):
        out = []
        for it in items:
            t0 = time.time()
            try:
                if it["study"] == "A":
                    text = self._render(it["prompt"])
                    sp = it.get("spans") or {}
                    positions, labels = [], []
                    for label, key in (("benign_email_end", "attacker_start"),
                                       ("attacker_email_end", "attacker_end")):
                        if sp.get(key) is None:
                            continue
                        positions.append(self._token_at_char(text, len(self.template_prefix) + sp[key] - 1)
                                         + self.pos_shift)
                        labels.append(label)
                    positions.append(-1)
                    labels.append("prompt_end")
                elif it["study"] == "C":
                    text = self.tok.apply_chat_template(it["messages"], add_generation_prompt=True,
                                                        tokenize=False)
                    turn = it["messages"][-1]["content"]
                    at = text.rfind(turn)
                    if at < 0:
                        raise ValueError("second turn not found in rendered conversation")
                    first = self._token_at_char(text, at)
                    last = self._token_at_char(text, at + len(turn) - 1)
                    positions = [p + self.pos_shift for p in range(first, last + 1, TURN_STRIDE)]
                    labels = [f"turn_token_{p - first - self.pos_shift}" for p in positions]
                    positions.append(-1)
                    labels.append("prompt_end")
                else:
                    text = self._render(it["prompt"], it["response"])
                    prompt_only = self._render(it["prompt"])
                    start = len(self.tok(prompt_only, add_special_tokens=False)["input_ids"])
                    n = len(self.tok(text, add_special_tokens=False)["input_ids"])
                    positions = [p + self.pos_shift for p in range(start, n, RESPONSE_STRIDE)]
                    labels = [f"answer_token_{p - start}" for p in positions]
                    if it.get("tool_call_char") is not None:
                        at = self._token_at_char(text, len(prompt_only) + it["tool_call_char"])
                        positions.append(at - 1 + self.pos_shift)
                        labels.append("just_before_tool_call")
                rows, n_tokens = self._read(text, positions, labels)
                out.append({"id": it["id"], "study": it["study"], "group": it["group"],
                            "base_id": it["base_id"], "outcome": it.get("outcome"),
                            "condition": it.get("condition"), "variant": it.get("variant"),
                            "success_rate": it.get("success_rate"), "tokens": n_tokens,
                            "seconds": round(time.time() - t0, 1), "readings": rows})
            except Exception as e:  # one bad item must not sink the chunk
                out.append({"id": it["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
                print(f"FAILED {it['id']}: {type(e).__name__}: {e}", flush=True)
        print(f"chunk of {len(items)} read", flush=True)
        return out


@app.local_entrypoint()
def main(limit: int = 0, chunk: int = 20, study: str = "", tag: str = "phi3-wikitext100",
         out: str = "", overwrite: bool = False, sample: str = ""):
    # --sample and --out are taken relative to the repo root, not to modal/
    sample_path = SAMPLE if not sample else (sample if os.path.isabs(sample) else os.path.join(ROOT, sample))
    items = [json.loads(line) for line in open(sample_path)]
    if study:
        items = [it for it in items if it["study"] == study]
    if limit:
        by = {}
        for it in items:
            by.setdefault((it["study"], it["group"], it.get("outcome"), it.get("condition"),
                           it.get("variant")), []).append(it)
        per = max(1, limit // len(by))
        items = [x for group in by.values() for x in group[:per]]

    out_path = OUT if not out else (out if os.path.isabs(out) else os.path.join(ROOT, out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "w"
    if os.path.exists(out_path) and not overwrite:
        done = {json.loads(l)["id"] for l in open(out_path)}
        items = [it for it in items if it["id"] not in done]
        mode = "a"
        print(f"resuming {out_path}: {len(done)} already read, {len(items)} to go")
    if not items:
        print("nothing left to read")
        return

    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    print(f"reading {len(items)} items in {len(chunks)} chunks with lens '{tag}'")
    reader = LensReader(tag=tag)
    t0, done = time.time(), 0
    with open(out_path, mode) as f:
        for res in reader.read.map(chunks, order_outputs=False, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"chunk FAILED remotely: {type(res).__name__}: {str(res)[:200]}")
                continue
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            done += 1
            print(f"chunk {done}/{len(chunks)} written ({time.time() - t0:.0f}s)")
    print(f"done -> {out_path}")
