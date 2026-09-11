"""Gradient probe as an HTTPS endpoint, so the Node harness can call it with fetch.
Deploy:  modal deploy probe_endpoint.py     (prints a URL)

POST <url>  JSON body:
  { "messages": [{"role":"system","content":...},{"role":"user","content":...}],
    "target": "send" }
Returns:
  { "target_prob": float, "top5": [str], "per_token": [[token, grad_norm], ...] }

Model stays loaded between requests while the container is warm (scaledown_window).
"""
import modal
from common import app, volume, MODEL_DIR, MODEL_PATH

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "accelerate", "fastapi[standard]")
    .add_local_python_source("common")  # Modal >=1.0 only uploads the entry file; ship common.py too
)


@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_DIR: volume},
    timeout=30 * 60,
    scaledown_window=10 * 60,
)
class Probe:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def probe(self, body: dict):
        import torch
        messages = body["messages"]
        target = body.get("target", "send")

        # Same chat template vLLM applies, so prompts match the benchmark exactly
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).cuda()
        target_id = self.tok(target, add_special_tokens=False).input_ids[0]

        emb = self.model.get_input_embeddings()(ids).detach().requires_grad_(True)
        out = self.model(inputs_embeds=emb)
        logit = out.logits[0, -1, target_id]
        logit.backward()

        grad_norms = emb.grad[0].float().norm(dim=-1).tolist()
        tokens = self.tok.convert_ids_to_tokens(ids[0])
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        return {
            "target_prob": probs[target_id].item(),
            "top5": [self.tok.decode(i) for i in probs.topk(5).indices.tolist()],
            "per_token": list(zip(tokens, grad_norms)),
        }
