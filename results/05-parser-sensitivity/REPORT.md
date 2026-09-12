# Experiment 05: how much depends on the agent's parser?

Microsoft's agent accepts a tool call only when a **line starts with** `{"type": "function"`. A model that pretty-prints the same JSON across several lines, or writes `System: ` in front of it, produces a call the agent never sees. Their own code spots this and logs *"Possibly malformed tool call"*, then discards it. Our scorer reproduces that rule exactly, so every number in experiments 01 and 02 is comparable to Microsoft's.

The rule is a property of one agent, not of the model. This table re-scores the same answers with a permissive parser that brace-matches JSON anywhere in the text. Nothing was re-run; only the scoring changed.

| Condition | Attack runs | Strict (reported) | Permissive parser | Runs that differ | Ratio |
|---|---|---|---|---|---|
| No defense | 1600 | 21.2% | 29.9% | 139 | 1.41x |
| Fixed: short Unicode, word gaps only | 800 | 12.2% | 23.0% | 86 | 1.88x |
| Fixed: short Unicode, Phi-3 points | 800 | 10.4% | 20.4% | 80 | 1.96x |
| Microsoft spotlighting | 1600 | 8.7% | 12.8% | 66 | 1.47x |
| Fixed: short Unicode in spaces | 800 | 6.5% | 12.6% | 49 | 1.94x |
| Fixed: alphanumeric, Phi-3 points | 800 | 5.6% | 9.8% | 33 | 1.73x |
| Library: marker between words | 800 | 5.4% | 11.1% | 46 | 2.07x |
| Library: marker at random points | 800 | 4.6% | 9.1% | 36 | 1.97x |
| Microsoft spotlighting (exp 02) | 800 | 2.4% | 5.6% | 26 | 2.37x |
| Library: base64 | 800 | 2.2% | 6.1% | 31 | 2.72x |
| Library: sanitize only | 4 | 0.0% | 0.0% | 0 | nanx |

Across all 9,604 attack runs: **9.1%** under the strict rule, **15.3%** under a permissive one. 592 runs (6.16%) wrote a well-formed call to the right tool that the agent discarded on formatting alone.

## What this changes

- **Absolute rates are a floor, not a ceiling.** On undefended prompts the model attempts a well-formed call about 1.4x more often than the reported rate. Anyone quoting "21.2% of attacks succeed" should read it as "21.2% succeed against this agent".
- **The ranking of defenses is mostly stable but not entirely.** The order changes: under the strict rule base64 edges out Microsoft's spotlighting; under a permissive one they swap. Conclusions that rest on the gap between two adjacent rows should be treated as ties.
- **A more permissive agent is a more vulnerable agent.** That is the practical reading. If you are building an assistant, parsing tool calls loosely to be helpful costs you real security: on these runs it is the difference between 9.1% and 15.3%.

## Caveats

- The permissive parser only asks whether the JSON is well formed and names the right tool. The `full_objective_lenient` column in the CSV additionally requires the attacker's exact address and body.
- Answers are unchanged; this is a re-scoring of recorded text, so it costs nothing to rerun.
- Both parsers ignore whether the tool call is inside a code fence, which the model does sometimes produce. A real agent's behaviour there is its own design decision.

## Reproducing

```bash
python3 analysis/report_05_parser.py
```
