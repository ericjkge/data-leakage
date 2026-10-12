# Question

Written 2026-10-11, before any run. Not to be edited after the first run. Changes go in `DECISIONS.md`.

## The setup in one paragraph

We take OLMo-7B and post-train it on documents about 70 fake people. The documents contain phone numbers, emails, ID numbers, addresses and health details, and are mixed with ordinary text so the model keeps its general ability. We then run the paper's attacks on it and look at what comes out:
- **Post-training data:** our planted documents.
- **Pretraining data:** the text OLMo was originally trained on.

This is done three times with different random seeds. Details are in `DECISIONS.md`; the procedure is in `PLAN.md`.

## Hypotheses

**H1: CIA still wins after post-training.** The paper found that CIA is the best of the attacks at getting a model to surface its pretraining data. We think that holds up on our post-trained model. CIA should beat the baselines (RSA, RWA, RA, EA) at surfacing both the new post-training data (the PANORAMA people) and the original pretraining data (Dolma).

Our guess for why: when CIA puts the model in its "confused" state, the context doesn't carry any useful information anymore. Conditioning on it is basically useless, so the model falls back on its unconditional distribution, what it would write with no prompt at all, and starts spewing training data.

**H2: the post-training data comes out more.** In our post-trained model, we think the post-training data will be spewed more often than the pretraining data. This follows from the same logic. If a confused model falls back on its unconditional distribution, and fine-tuning shifts that distribution toward the fine-tuning data, then the fine-tuning data should be what comes out.

**If we have time:** we can check the "falls back on the unconditional distribution" part directly. If it's right, the model's next-token predictions after a CIA prompt should look about the same as after an empty prompt, much more so than after a random prompt (RSA).

## What would prove each one wrong

- **H1 is wrong** if, for either kind of data, any of the four other attacks surfaces it at least as often as CIA. That is, the test in D-32 fails to show CIA ahead of all four. The two halves are reported separately.
- **H2 is wrong** for an attack if its generations contain pretraining data at least as often as post-training data. Each attack is reported separately.
- **The reasoning is wrong** (if we run the check) if CIA prompts change the model's predictions as much as random prompts do, or more.

## What we expect, written down so it can't move

- **H1, pretraining half: we expect it to fail.** Our reproduction on the base model already has CIA third of five. These are rates of 50-token exact matches, 500 prompts each:

  | Attack | Rate |
  |---|---|
  | RWA | 8.2% |
  | RSA | 5.6% |
  | CIA | 2.0% |
  | RA | 1.2% |
  | EA | 0.0% |

- **H1, post-training half:** genuinely open.
- **H2: we expect it to hold,** but repetition and recency alone would predict that too, so it holding won't confirm our explanation (D-25, point 5).

The reported result is whatever comes out, either way.
