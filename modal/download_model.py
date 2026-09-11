"""Step 1: one-off download of Phi-3-medium into the Modal volume.
Run:  modal run download_model.py
"""
import modal
from common import app, volume, serve_image, HF_SECRETS, MODEL_ID, MODEL_DIR, MODEL_PATH

REVISION = "main"  # replace with a commit hash from the HF repo's Files tab


@app.function(
    image=serve_image,
    volumes={MODEL_DIR: volume},
    secrets=HF_SECRETS,
    timeout=60 * 60,   # 28 GB download, give it an hour
)
def download():
    from huggingface_hub import snapshot_download
    # revision pinned for reproducibility; bump deliberately, never silently
    snapshot_download(MODEL_ID, local_dir=MODEL_PATH, revision=REVISION)
    volume.commit()
    import os
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(MODEL_PATH) for f in fs)
    return f"downloaded {total/1e9:.1f} GB to {MODEL_PATH}"


@app.local_entrypoint()
def main():
    print(download.remote())
