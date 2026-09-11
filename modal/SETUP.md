# Setup walkthrough (do these in order)

## A. Hugging Face token — YOU DO NOT NEED ONE

Verified on 2026-09-11 against the Hub API:

    microsoft/Phi-3-medium-128k-instruct  ->  gated: False, private: False, license: mit

An anonymous download returns HTTP 200. The code defaults to sending NO token.
Fewest credentials = smallest blast radius. Skip to Part B.

### Only if you hit HTTP 429 (rate limit) during download

Do NOT create a `read` token. A `read` token can pull every private repo you or
any org you belong to can see. Create a **fine-grained** token instead:

1. https://huggingface.co/settings/tokens -> **Create new token**
2. Token type: **Fine-grained**
3. Name: `modal-phi3-readonly`
4. Under **Repositories**: search `microsoft/Phi-3-medium-128k-instruct`,
   select it, and tick ONLY `Read access to contents of selected repos`.
5. Leave EVERY box under **User permissions** unticked. Especially these:
   - Write access to contents/settings
   - Manage Inference Endpoints
   - Make calls to Inference Providers   <- this one can spend your money
   - Manage billing
   - Manage webhooks / collections / discussions
6. Under **Organizations**: select nothing.
7. Create, copy the `hf_...` value. Shown once.
8. Modal dashboard -> Secrets -> name `huggingface-secret`, key `HF_TOKEN`.
9. Then run downloads with:  `USE_HF_SECRET=1 modal run download_model.py`

Rotate or kill a leaked token at https://huggingface.co/settings/tokens (Manage ->
Delete/Refresh). Anyone can revoke a leaked token they find, no account access needed:

    curl -X POST "https://huggingface.co/api/credentials/revoke" \
      -H "Content-Type: application/json" \
      -d "{\"credentials\": [\"$LEAKED\"]}"

## B. Modal account

1. Sign up: https://modal.com/signup  (use GitHub or Google login)
2. You get a workspace automatically. Note its name, it shows in the top-left of the dashboard
   and becomes part of your endpoint URLs.

### B1. Spending limit (DO THIS FIRST)

1. Go to: https://modal.com/settings  -> **Usage & Billing** (or **Plan**)
2. Set a spend limit. Suggested: **$50**.
3. Turn on email alerts at 50% and 80%.

### B2. HF token as a Secret

1. Go to: https://modal.com/secrets
2. Click **Create new secret**.
3. Choose the **Hugging Face** template if offered, otherwise **Custom**.
4. Secret name: `huggingface-secret`   (exact, the code looks for this name)
5. Key: `HF_TOKEN`     Value: the fine-grained `hf_...` token from Part A.
6. Click **Create**.

### B3. Proxy auth token (protects your GPU from strangers)

1. Go to: https://modal.com/settings -> **Proxy Tokens**
2. Click **New Token**. Name it `llmail-harness`.
3. Copy BOTH values:
   - Token ID     -> goes in the `Modal-Key` header
   - Token Secret -> goes in the `Modal-Secret` header, shown ONCE
4. Keep them for your `.env`.

## C. Local CLI (uv)

`uv` replaces pip and virtualenv. It is much faster and keeps your system Python clean.

### C0. Confirm uv is installed

    uv --version

If that errors, install it:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then restart your terminal, or run `source $HOME/.local/bin/env`.

### C1. Install the Modal CLI

Modal is a tool you run, not a library you import into this project, so install it
as a standalone tool. `uv` gives it its own isolated environment automatically:

    uv tool install modal

Verify:

    modal --version

If `modal` is not found, run `uv tool update-shell` once and restart the terminal.

Alternative, if you would rather not install anything: prefix every command with
`uvx`, which downloads and runs it in a throwaway environment.

    uvx modal setup
    uvx modal run download_model.py

The rest of this guide writes plain `modal ...`. Add `uvx ` in front if you chose that route.

### C2. Authenticate

    modal setup

Opens a browser tab. Click approve. It writes `~/.modal.toml`.
Check it worked:

    modal profile current

### C3. Project environment for local analysis (optional, do later)

You only need this when you start analysing results locally with pandas/matplotlib.
From the repo root:

    uv init --bare          # creates pyproject.toml, does not touch your Node files
    uv add pandas pyarrow matplotlib numpy
    uv run python analysis.py

`uv run` auto-creates and syncs `.venv` for you. Never `pip install` into it by hand.
Add `.venv` to `.gitignore`.

## D. Download the weights (one-off, ~20 min, no GPU cost)

    cd modal
    modal run download_model.py

Watch the log. It ends with `downloaded 28.x GB to /models/phi3-medium`.
Verify it stuck around:

    modal volume ls phi3-weights

## E. Poke around before deploying (cheap debugging)

    modal shell --gpu A100-80GB download_model.py::download

Drops you in a terminal on a GPU container with the volume mounted.
`ls /models/phi3-medium` should list the weight files. Type `exit` when done.

## F. Deploy the two endpoints

    modal deploy serve_vllm.py        # prints URL A
    modal deploy probe_endpoint.py    # prints URL B

URLs look like: https://<workspace>--phi3-injection-serve.modal.run
You can always find them again at https://modal.com/apps

## G. Wire up the Node harness

Add to `.env` in the repo root (NOT in git, check .gitignore):

    LMSTUDIO_BASE_URL=<URL A>/v1
    LMSTUDIO_MODEL=phi3-medium
    LMSTUDIO_API_KEY=unused
    LMSTUDIO_TIMEOUT_MS=300000
    PROBE_URL=<URL B>
    MODAL_KEY=<Proxy Token ID>
    MODAL_SECRET=<Proxy Token Secret>

Every request to either URL needs these two headers:

    'Modal-Key': process.env.MODAL_KEY,
    'Modal-Secret': process.env.MODAL_SECRET,

Test URL A from the terminal (first call takes ~2 min, vLLM is cold-starting):

    curl "$LMSTUDIO_BASE_URL/chat/completions" \
      -H "Modal-Key: $MODAL_KEY" \
      -H "Modal-Secret: $MODAL_SECRET" \
      -H "Content-Type: application/json" \
      -d '{"model":"phi3-medium","messages":[{"role":"user","content":"Say hi"}]}'

## H. Shut things down when idle

    modal app list
    modal app stop phi3-injection

Containers also stop themselves after `scaledown_window`. The Volume keeps costing
a few cents a month for 28 GB, which is fine. Delete it only when fully done:

    modal volume delete phi3-weights

## Safety checklist

- [ ] Spend limit set before first deploy
- [ ] `requires_proxy_auth=True` on both endpoints (already in the code)
- [ ] `.env` is in `.gitignore`, tokens never committed
- [ ] HF token is **Read** scope only
- [ ] `modal app stop` after a work session

---

# Security advisory

## Credential inventory — what exists and what it can do

| Credential | Needed? | If leaked, attacker can | Blast radius control |
|---|---|---|---|
| HF token | **No** (Phi-3 is public) | Read your private repos; a fine-grained one scoped to 1 repo can do almost nothing | Just don't create one |
| Modal Proxy Token ID + Secret | Yes | Call your GPU endpoints and burn your credits | Spend limit + rotate token |
| Modal account login | Yes | Full workspace control | 2FA on GitHub/Google account |

## The real risks, ranked

1. **Money, not data.** Your biggest exposure is a stranger finding an unauthenticated
   GPU endpoint and mining on your A100. `requires_proxy_auth=True` is set on both
   endpoints. Never comment it out "just to test quickly" — that is exactly how these leak.
   Second line of defense is the spend limit. Set it before the first deploy.

2. **Modal secrets are not a vault.** Values are readable in the dashboard by anyone
   with workspace access. Fine for a solo workspace. Do not invite collaborators and
   then assume the token is hidden from them.

3. **Logs leak.** Modal streams function logs to the dashboard. Never `print()` a token,
   a header dict, or `os.environ`. Your probe returns token strings from the email text,
   which is fine, but do not add debug prints of request headers.

4. **Git history is forever.** `.env` is gitignored here — I checked. But if a token ever
   lands in a commit, removing it in a later commit does NOT remove it. Rotate the token,
   don't just delete the line.

5. **You are running attack payloads on purpose.** The LLMail dataset is 462k real
   injection attempts. Two rules:
   - Your harness must only ever **parse** tool calls, never execute them. I checked
     `src/toolParser.js` and `src/runner.js`: parse-only, no `exec`/`spawn`/`eval`. Keep it that way.
   - Never point this harness at a real mail account, a real MCP server with real tools,
     or anything with side effects. The whole point is that some of these payloads work.

6. **Model output is untrusted input.** Anything the model returns was influenced by an
   attacker-authored email. Do not feed its output into a shell, a file path, or another
   agent without validation.

## Hardening checklist (tick before first deploy)

- [ ] Modal spend limit set ($50) with email alerts
- [ ] 2FA on the GitHub/Google account used for Modal login
- [ ] No HF token created (or fine-grained, single-repo, read-only)
- [ ] `requires_proxy_auth=True` present in serve_vllm.py and probe_endpoint.py
- [ ] Proxy token secret stored in `.env`, and `.env` is gitignored
- [ ] `git log -p | grep -i "hf_\|modal-secret"` returns nothing
- [ ] Harness has no tool-execution path
- [ ] `modal app stop phi3-injection` after each session

## If something leaks

1. HF token: https://huggingface.co/settings/tokens -> Manage -> Delete.
2. Modal proxy token: https://modal.com/settings -> Proxy Tokens -> revoke, create new.
3. Then `modal app stop phi3-injection` and check https://modal.com/settings usage
   for unexpected GPU seconds.
