# Vendored reference files from microsoft/llmail-inject-challenge

Copied verbatim, unmodified, from commit `ad115315c1cb34381d20875d6675a6cfe6ca80fa`
(https://github.com/microsoft/llmail-inject-challenge). MIT License, see LICENSE.

| File | Original path | Used for |
|---|---|---|
| config.yaml | src/agent/config.yaml | system prompt, tool prompt, scenario1 query, top_p, max_new_tokens |
| level1.json | src/agent/data/level1.json | the benign email that sits beside the attacker email in level 1 |
| fp_tests.json | src/agent/data/fp_tests.json | clean emails for the later utility check |
| prompt_utils.py | src/agent/workloads/prompt_utils.py | spotlighting strings, parsed with ast, never imported |
| email_retriever.py | src/agent/workloads/email_retriever.py | special-token list, parsed with ast, never imported |

Do not edit these. `../llmail_prompt.py` reads them so the prompts match the challenge byte for byte.
