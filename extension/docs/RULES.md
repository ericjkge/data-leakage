# Rules for the extension

These apply to everything in `extension/`, and to any change to shared code (`src/`, `scripts/`, `configs/`) made for the extension.

### 1. Every file has a reason.
If you can't write a one-sentence reason for a file, don't create it. This applies to data, code, figures and documents alike. The current files and their reasons:

| File | Why it exists |
|---|---|
| `RUN.md` | The commands for whoever runs the GPU pod, and nothing else |
| `docs/QUESTION.md` | What we're testing and what would prove it wrong, written before any run |
| `docs/PLAN.md` | The procedure: what gets built and run, in what order, how long it takes |
| `docs/FINDINGS.md` | The results so far in short, each pointing to the DECISIONS entry with the full numbers |
| `docs/DECISIONS.md` | Every design decision with its reason, written before the run it affects |
| `docs/INHERITANCE.md` | Everything reused from outside `extension/`, with the command that checks it |
| `docs/RULES.md` | How we work in this folder |
| `pipeline/setup_pod.sh` | One-time pod setup: Python environment with the tested versions, plus the model download |
| `pipeline/run_training.sh` | The one command that trains all three seeds on the pod |
| `pipeline/fetch_dolma.py` | Downloads the shared Dolma text once, so seeds don't each download it |
| `pipeline/build_data.py` | Builds each seed's training text: draws the people, picks their documents, mixes them into the Dolma text |
| `pipeline/pii.py` | The one copy of the rules for spotting an identifier or health detail, used to build the data and to score leaks |
| `pipeline/train.py` | Post-trains one seed with LoRA, saving checkpoints it can resume from |
| `pipeline/checks.py` | The four checks before any attack (D-19, D-52) |
| `pipeline/run_attacks.sh` | The one command that runs every attack on the pod (step 4, D-53) |
| `pipeline/score.py` | Finds leaks and runs the tests (steps 5–6, D-54) |
| `pipeline/audit.py` | Builds the blind hand-check sheet and reads the labels back (step 5, D-31, D-59) |
| `pipeline/figures.py` | The three presentation figures, from committed results only (writes `results/figures/`) |
| `audit/` | The hand-check sheet, its hidden key, the labels and the labellers' agreement |
| `results/` | Leak rates, the by-copies table and the H1/H2 tests, written by `score.py tables` |
| `selection/dolma.json` | Which Dolma documents were picked, plus a fingerprint, so anyone can check a rebuild |
| `selection/seed_s/people.json` | Seed s's 100 people, groups, identifiers and documents, so the build needs no HF account |
| `selection/seed_s/build.json` | Seed s's token counts, private share, and the fingerprint of its training text |
| `checks/` | Results of the four checks: one small JSON file per check and seed |
| `models/` | Training output: small run records (committed) and adapter weights (not committed; copied to HF) |
| `data/` (not committed) | Downloads and built training text; all of it can be rebuilt |

**Layout:** `RUN.md` at the top for whoever runs the GPU, the documents in `docs/`, all code in `pipeline/`, the committed record of what each seed trains on in `selection/`, and training output in `models/`.

### 2. Every design decision has a reason, written before the run it justifies.
This matters more than rule 1. A choice of dataset size, learning rate, threshold, measure or comparison is a decision. It goes in `DECISIONS.md` with its reason and the condition that would make us revisit it. A decision recorded after seeing results is not a decision, it's an explanation. This holds for everyone: whoever changes the design adds the entry.

### 3. Nothing is inherited on trust.
Anything reused from outside `extension/` goes in `INHERITANCE.md` with the command that verifies it and the actual output of running that command. That covers:
- our reproduction's code and results
- PANORAMA-Plus
- Dolma
- the infini-gram service

Anything that can't be independently checked is rebuilt, not assumed.

### 4. The hypothesis is written before the result.
`QUESTION.md` states what is being tested and what outcome would prove it wrong. It is written before any run. This is what makes a negative result worth reporting rather than embarrassing.

### 5. Simple and explainable beats thorough.
Better to run something understandable and get a poor result than to run something complicated that nobody can explain. When a check, a measure or a design grows a special case to patch a loophole, delete the special case and accept the looser result.

### 6. Coverage is not the goal.
The extension does not redo the reproduction or add every analysis we can think of. It does what `QUESTION.md` needs, plus whatever is needed to support its own claims. Anything else is cut (`DECISIONS.md` D-38 lists what has been cut so far).

### 7. Be skeptical of the reasoning underneath, not just the result.
For any method we rely on, including the paper's, ask what it assumes and where it could be wrong. Write the weak points down even when we go ahead anyway (D-25 does this for our own hypotheses). An unexamined method that happens to work is still a liability.

### 8. No jargon.
Explanations are written so that someone who hasn't been in the conversation can follow them. If a term is unavoidable, define it where it first appears. "Seed", "held out", "post-training data" and "VM@50" are all defined in `DECISIONS.md` the first time they're used.

### 9. Declare limitations up front.
Known weaknesses go in the writeup before a grader finds them. A stated limitation is evidence of understanding; an unstated one that gets discovered is evidence of the opposite. The current list is D-39.

### 10. Whoever presents must be able to defend every number.
If a result can't be explained without the agent that produced it, it doesn't belong in the submission.
