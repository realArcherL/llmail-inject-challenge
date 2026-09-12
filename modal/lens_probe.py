"""Experiment 07: where does a Jacobian lens on Phi-3 actually become readable?

Experiment 03 claimed the lens decodes from "about layer 20" on the basis of FIVE layers probed with
TWO facts. That is not enough to support a claim about the shape of the readable band, so this
measures it properly and tries to break it.

For every layer 0..38, every fact, and every context depth it records where the correct answer sits
in the lens readout -- its rank and its probability -- not merely whether it appears in a top-5 list.
A rank is a graded measure, so a gradual emergence and a sharp threshold look different.

Things that could make the claim wrong, and how each is addressed:
  too few facts          -> 30 facts across geography, science, language, arithmetic and idiom
  too few layers         -> every layer, no interpolation across gaps
  one context length     -> four depths, from 10 tokens to past our real prompts
  tokenizer artefact     -> the answer is scored with and without a leading space, best rank kept
  one lens fit           -> runs against both independently fitted lenses
  "readable" is a guess  -> reports rank, probability and the model's own rank for comparison
  prompt phrasing        -> each fact is asked in two different framings

Writes runs/07-layer-probe/probes.jsonl
Run:  modal run lens_probe.py                       # both lenses, all layers
      modal run lens_probe.py --tags phi3-wikitext100
"""
import json
import os
import time

import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENS_DIR = f"{MODEL_DIR}/lens"
LENS_REPO = "https://github.com/anthropics/jacobian-lens.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch", "transformers>=5.5", "accelerate", "jinja2", "numpy")
    .run_commands(f"git clone --depth 1 {LENS_REPO} /opt/jlens", "cd /opt/jlens && pip install -e .")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("common", "hfload")
)

# (name, category, question A, question B, answer). Two framings each, so a result cannot rest on
# one turn of phrase. Answers are single words the tokenizer keeps whole.
FACTS = [
    ("france", "geography", "Fact: the capital city of France is", "The capital of France is the city of", "Paris"),
    ("japan", "geography", "Fact: the capital city of Japan is", "The capital of Japan is the city of", "Tokyo"),
    ("italy", "geography", "Fact: the capital city of Italy is", "The capital of Italy is the city of", "Rome"),
    ("egypt", "geography", "Fact: the longest river in Egypt is the", "Egypt's longest river is called the", "Nile"),
    ("everest", "geography", "Fact: the highest mountain on Earth is Mount", "The tallest mountain in the world is Mount", "Everest"),
    ("yen", "economics", "Fact: the currency used in Japan is the", "People in Japan pay for things using the", "yen"),
    ("euro", "economics", "Fact: the currency used in Germany is the", "People in Germany pay for things using the", "euro"),
    ("water", "science", "Fact: the chemical formula for water is H2", "Water's chemical formula is written H2", "O"),
    ("gold", "science", "Fact: the chemical symbol for gold is", "On the periodic table gold is written as", "Au"),
    ("planets", "science", "Fact: the closest planet to the Sun is", "The planet nearest the Sun is called", "Mercury"),
    ("gravity", "science", "Fact: things fall because of the force called", "The force that pulls objects down is", "gravity"),
    ("photosynth", "science", "Fact: plants make food using a process called", "The process plants use to make food is", "photosynthesis"),
    ("twoplus", "arithmetic", "Fact: two plus three equals", "If you add two and three you get", "5"),
    ("sevenplus", "arithmetic", "Fact: seven plus eight equals", "If you add seven and eight you get", "15"),
    ("dozen", "arithmetic", "Fact: the number of items in a dozen is", "A dozen contains this many items:", "12"),
    ("sides", "arithmetic", "Fact: the number of sides on a triangle is", "A triangle has this many sides:", "3"),
    ("french", "language", "Fact: the word for 'hello' in French is", "In French, people greet each other by saying", "bonjour"),
    ("spanish", "language", "Fact: the word for 'water' in Spanish is", "In Spanish, water is called", "agua"),
    ("opposite", "language", "Fact: the opposite of 'hot' is", "The word that means the opposite of hot is", "cold"),
    ("plural", "language", "Fact: the plural of 'mouse' is", "More than one mouse is called", "mice"),
    ("shakespeare", "culture", "Fact: the author of Romeo and Juliet was William", "Romeo and Juliet was written by William", "Shakespeare"),
    ("mona", "culture", "Fact: the Mona Lisa was painted by Leonardo da", "The Mona Lisa is a painting by Leonardo da", "Vinci"),
    ("chess", "culture", "Fact: the number of squares on a chessboard is", "A chessboard has this many squares:", "64"),
    ("rainbow", "common", "Fact: the colour of a clear daytime sky is", "On a clear day the sky looks", "blue"),
    ("snow", "common", "Fact: the colour of fresh snow is", "Fresh snow is coloured", "white"),
    ("week", "common", "Fact: the number of days in a week is", "A week contains this many days:", "7"),
    ("legs", "common", "Fact: the number of legs on a spider is", "A spider has this many legs:", "8"),
    ("cow", "common", "Fact: the sound a cow makes is a", "Cows make a sound called a", "moo"),
    ("fire", "common", "Fact: the thing firefighters use to put out fires is", "Firefighters put out fires using", "water"),
    ("doctor", "common", "Fact: the person you see when you are ill is a", "When you feel ill you visit a", "doctor"),
]
FILLER = ("The river system supports a mix of farms, small towns and light industry along its banks. "
          "Rainfall arrives mostly in spring, and the water level falls through the dry months. "
          "Local councils have argued about the bridge for years without reaching a decision. ")
DEPTHS = [("short", 0), ("mid_400", 6), ("mid_1200", 20), ("deep_3300", 60)]


@app.cls(image=image, gpu="A100-80GB", volumes={MODEL_DIR: volume}, memory=65536,
         timeout=90 * 60, max_containers=4, scaledown_window=60)
class Prober:
    @modal.enter()
    def load(self):
        import jlens, torch, hfload
        cfg = hfload.load_config(MODEL_PATH)
        self.tok = hfload.load_tokenizer(MODEL_PATH, cfg)
        self.hf = hfload.load_model(MODEL_PATH, cfg)
        self.model = jlens.from_hf(self.hf, self.tok, force_bos=False)
        self.jlens, self.torch = jlens, torch
        volume.reload()

    def _answer_ids(self, answer):
        """Both spellings of the answer: with and without a leading space."""
        ids = []
        for form in (answer, " " + answer):
            enc = self.tok(form, add_special_tokens=False)["input_ids"]
            if enc:
                ids.append(int(enc[0]))
        return sorted(set(ids))

    @modal.method()
    def probe(self, tag, facts):
        lens = self.jlens.JacobianLens.load(f"{LENS_DIR}/{tag}.pt")
        out = []
        for name, category, qa, qb, answer in facts:
            ids = self._answer_ids(answer)
            for framing, question in (("A", qa), ("B", qb)):
                for depth, reps in DEPTHS:
                    text = (FILLER * reps + " " + question) if reps else question
                    t0 = time.time()
                    lens_logits, model_logits, tok_ids = lens.apply(
                        self.model, text, positions=[-1], max_seq_len=4096)
                    final = model_logits.reshape(-1, model_logits.shape[-1])[0].float()
                    fs = self.torch.softmax(final, dim=-1)
                    frank = int(min((final > final[i]).sum().item() for i in ids)) + 1
                    rec = {"fact": name, "category": category, "framing": framing, "depth": depth,
                           "answer": answer, "answer_ids": ids, "tag": tag,
                           "tokens": int(tok_ids.shape[-1]) if hasattr(tok_ids, "shape") else len(tok_ids),
                           "model_rank": frank, "model_prob": round(float(max(fs[i] for i in ids)), 6),
                           "model_top": [self.tok.decode([i]) for i in final.topk(5).indices.tolist()],
                           "layers": {}, "seconds": round(time.time() - t0, 1)}
                    for layer, lg in sorted(lens_logits.items()):
                        v = lg.reshape(-1, lg.shape[-1])[0].float()
                        p = self.torch.softmax(v, dim=-1)
                        rank = int(min((v > v[i]).sum().item() for i in ids)) + 1
                        rec["layers"][int(layer)] = {
                            "rank": rank,
                            "prob": round(float(max(p[i] for i in ids)), 8),
                            "top": [self.tok.decode([i]) for i in v.topk(5).indices.tolist()],
                        }
                    out.append(rec)
            print(f"{tag} {name} done", flush=True)
        return out


@app.local_entrypoint()
def main(tags: str = "phi3-wikitext100,phi3-wikitext100b", chunk: int = 8, out: str = ""):
    out_path = out or os.path.join(ROOT, "runs", "07-layer-probe", "probes.jsonl")
    out_path = out_path if os.path.isabs(out_path) else os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    jobs = []
    for tag in [t.strip() for t in tags.split(",") if t.strip()]:
        for i in range(0, len(FACTS), chunk):
            jobs.append((tag, FACTS[i:i + chunk]))
    print(f"probing {len(FACTS)} facts x 2 framings x {len(DEPTHS)} depths x every layer, "
          f"for {len(tags.split(','))} lens(es): {len(jobs)} jobs")
    prober, n, t0 = Prober(), 0, time.time()
    with open(out_path, "w") as f:
        for res in prober.probe.starmap(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(res, BaseException):
                print(f"job FAILED: {type(res).__name__}: {str(res)[:200]}")
                continue
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            print(f"job {n}/{len(jobs)} ({time.time() - t0:.0f}s)")
    print(f"done -> {out_path}")
