"""Model loading that works on transformers >= 5.5, which jlens requires.

Phi-3-medium-128k's config.json was written for transformers 4.39. Newer
transformers moved RoPE settings into `rope_parameters` and now insists that
longrope ("su") scaling carries `original_max_position_embeddings` inside them.
The file has that key, just one level up. We copy it down in memory only.
Files on the Modal volume are never modified.

No modal import here on purpose, so this can be tested on a laptop.
"""
import json
import os


def _config_json_path(model_path):
    if os.path.isdir(model_path):
        return os.path.join(model_path, "config.json")
    from huggingface_hub import hf_hub_download  # hub id, e.g. the Qwen fallback
    return hf_hub_download(model_path, "config.json")


def load_config(model_path):
    import transformers
    with open(_config_json_path(model_path)) as f:
        raw = json.load(f)
    rs = raw.get("rope_scaling")
    if (isinstance(rs, dict)
            and rs.get("type", rs.get("rope_type")) in ("su", "longrope")
            and "original_max_position_embeddings" not in rs
            and "original_max_position_embeddings" in raw):
        rs["original_max_position_embeddings"] = raw["original_max_position_embeddings"]
    return transformers.CONFIG_MAPPING[raw["model_type"]].from_dict(raw)


def load_tokenizer(model_path, config=None):
    # Must receive the patched config, or it re-reads config.json and crashes.
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, config=config or load_config(model_path))


def load_model(model_path, config=None, device_map="cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config or load_config(model_path),
        dtype=torch.bfloat16, device_map=device_map)
    return model.eval()
