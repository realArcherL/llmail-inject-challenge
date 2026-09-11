"""Step 2: serve Phi-3-medium behind an OpenAI-compatible HTTPS endpoint.
Deploy: modal deploy serve_vllm.py
Then point your Node harness at the printed URL + /v1, model name "phi3-medium".
"""
import subprocess
import modal
from common import app, volume, serve_image, MODEL_DIR, MODEL_PATH

PORT = 8000
SERVED_NAME = "phi3-medium"


@app.function(
    image=serve_image,
    gpu="L40S",                  # 48 GB, enough for 14B bf16 inference. Use "A100-80GB" for long contexts.
    volumes={MODEL_DIR: volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,     # container stops 5 min after last request; cold start ~2 min
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=10 * 60, requires_proxy_auth=True)
def serve():
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "16384",   # challenge emails are short; raise if needed
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
