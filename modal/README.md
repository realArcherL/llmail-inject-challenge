# Modal box for Phi-3-medium

Order of operations:

1. `pip3 install modal && modal setup`
2. Create Modal secret `huggingface-secret` with key `HF_TOKEN` (dashboard -> Secrets).
3. `modal run download_model.py`        # one-off, 28 GB into volume "phi3-weights"
4. `modal deploy serve_vllm.py`         # prints URL A  (OpenAI-compatible, L40S)
5. `modal deploy probe_endpoint.py`     # prints URL B  (gradient probe, A100)
6. `modal run jacobian_probe.py`        # sanity check without deploying anything

Node harness `.env`:

    LMSTUDIO_BASE_URL=<URL A>/v1
    LMSTUDIO_MODEL=phi3-medium
    LMSTUDIO_API_KEY=anything
    LMSTUDIO_TIMEOUT_MS=300000          # first call includes ~2 min cold start
    PROBE_URL=<URL B>

Calling the probe from JS:

    const r = await fetch(process.env.PROBE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, target: 'send' }),
    });
    const { target_prob, top5, per_token } = await r.json();

`messages` is the same array you already send to /chat/completions,
so the probe sees exactly the prompt the benchmark saw.
