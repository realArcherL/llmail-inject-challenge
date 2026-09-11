"""Shared Modal config for the Phi-3 box."""
import modal

MODEL_ID = "microsoft/Phi-3-medium-128k-instruct"
MODEL_DIR = "/models"          # volume mount point inside containers
MODEL_PATH = f"{MODEL_DIR}/phi3-medium"

app = modal.App("phi3-injection")

# Persistent disk. Weights go here once and survive across runs.
volume = modal.Volume.from_name("phi3-weights", create_if_missing=True)

# Phi-3-medium-128k-instruct is public, ungated, MIT. It downloads anonymously.
# NO Hugging Face token is required. Leave HF_SECRETS empty (the secure default).
#
# Only if you hit HTTP 429 rate limits: create a FINE-GRAINED token scoped to
# read exactly this one repo, store it as a Modal secret named
# "huggingface-secret" with key HF_TOKEN, then set the env var below:
#     export USE_HF_SECRET=1
import os

HF_SECRETS = (
    [modal.Secret.from_name("huggingface-secret")]
    if os.environ.get("USE_HF_SECRET")
    else []
)

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "accelerate", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
