# Experiment 08: the one thing the lens sees that the output does not

Everywhere else in this project, reading the model's own next-token distribution beats reading the Jacobian lens (see `REPORT-baseline.md`). This is the exception.

At the **end of the attacker's email** — mid-prompt, where a next-token distribution is not a useful thing to read — the lens sometimes reads turn-boundary tokens: `END`, `USER`, `SY`. That is the model representing the attack as though a speaking turn had ended and something else were about to talk, which is structurally what an indirect injection is: text impersonating a new speaker.

Measured on all **3,999** repeat-submitted undefended attacks, labelled with Microsoft's own recorded outcomes.

## Does it predict which attacks succeed?

| Reading at the end of the attack | Attacks showing it | Their win rate | Attacks not showing it | Their win rate | Intervals separate |
|---|---|---|---|---|---|
| turn_boundary | 660 | 29.6% | 3339 | 15.8% | yes |
| instruction_words | 1145 | 16.2% | 2854 | 18.9% | no |

The turn boundary separates cleanly: 29.6% (27.0% to 32.3%) against 15.8% (14.9% to 16.8%).

**The instruction words do not.** `ignore` / `instruction` / `warning` appear for 1,145 attacks and none of the 203 clean prompts, so they do distinguish an attack from an ordinary email — but they say nothing about which attacks work (16.2% (14.5% to 17.9%) against 18.9% (17.8% to 20.0%), and slightly the wrong way round). An earlier draft of this work highlighted those words. It should not have.

## Does it add anything the output does not already say?

Crossed against the model's own output flag at the end of the prompt, thresholded at the largest reading any clean prompt produced (0.00113):

| Group | Attacks | Win rate | 95% low | 95% high |
|---|---|---|---|---|
| output flag + boundary | 479 | 34.1% | 30.9% | 37.2% |
| output flag only | 1521 | 23.7% | 22.2% | 25.3% |
| boundary only | 181 | 17.9% | 13.7% | 22.1% |
| neither | 1818 | 9.3% | 8.3% | 10.3% |

And within each stratum separately, which is the test that matters:

| Among attacks that are… | With boundary | Win rate | Without | Win rate | Intervals separate |
|---|---|---|---|---|---|
| already flagged by the output | 479 | 34.1% | 1521 | 23.7% | yes |
| not flagged by the output | 181 | 17.9% | 1818 | 9.3% | yes |

**It adds information in both strata.** The boundary reading is not a restatement of the output signal; the two are read at different positions and each contributes.

## Where in the network it lives

Full table in `tables/boundary-by-layer.csv`. The reading appears from about layer 25 and separates at almost every layer from there to 38. Below 25 it barely occurs.

## What this does and does not show

- It is correlational. The reading does not cause the attack to succeed; both may follow from something about the attack's construction.
- Win rates are Microsoft's, from a median of two submissions per attack, so each label is coarse. With 3,999 attacks that averages out, but no single row means much.
- It does not show the model judging the email as illegitimate. Representing text as a turn boundary is a structural fact, not an opinion about it.
- One model, one prompt format. See the scope note in every other report here.

## Reproducing

```bash
python3 analysis/report_08_boundary.py
```
