# Running the extension on a GPU

For whoever runs the GPU pod. This page is only the commands and what to look for; the reasons are in [docs/](docs/). Right now it covers **training** (about 2–3 hours on one H100). The later steps (checks, attacks) will be added here when they're ready.

**You need:**
- A pod with one **H100 80 GB** (an A100 80 GB also works, just slower) and **at least 80 GB of disk**.
- **Optional:** a HuggingFace account, to back up the trained models (step 3).

All commands are run from the repo folder on the pod.

---

## 1. Get the code
```
cd /workspace
git clone https://github.com/ericjkge/data-leakage.git
cd data-leakage
```
If the repo is already on the pod, run `cd /workspace/data-leakage && git pull` instead.

## 2. Set up (once per pod, ~20–30 min, mostly downloading the 27.6 GB model)
```
bash extension/pipeline/setup_pod.sh
```
It should end with something like:
```
GPU: NVIDIA H100 80GB HBM3, 80 GB; torch 2.11.0+cu128, transformers 5.19.0, peft 0.21.2
setup done
```
If it says `no GPU visible` or `needs an 80 GB GPU`, stop and message Ishaan.

## 3. Optional: back up the trained models to HuggingFace
Without this, the trained models exist only on the pod until step 6.
1. Make a token with **Write** access at https://huggingface.co/settings/tokens.
2. Run these two commands:
   ```
   .venv/bin/hf auth login
   export HF_REPO=<your-hf-username>/olmo-pano-adapters
   ```
   Paste the token into the `hf auth login` prompt (not into any chat), and answer `n` if it asks about git credentials.

`HF_REPO` must be set in the same terminal as step 4.

## 4. Start training (all three seeds, ~1.5–2.5 h)
```
nohup bash extension/pipeline/run_training.sh > /dev/null 2>&1 &
tail -f extension/models/training.log
```
It keeps running if you close the terminal. `Ctrl-C` only stops `tail` (the log display); reopen the log with the same `tail -f` line.

## 5. Check it's working (first ~5–10 minutes)
In the log you should see, in order:
1. `=== trial: 5 steps of seed_0`, then five lines like:
   ```
   {"step": 1, "of": 157, ..., "loss_all": 2.4, "loss_private": 2.9, "loss_dolma": 2.3, ..., "tokens_per_s": 9000, "gpu_mem_gb": 38.0}
   ```
   Check that:
   - `"of"` is about **157**
   - the losses are between roughly **1.5 and 4**. Around 10.8 would mean the model didn't load properly; stop and message Ishaan.
   - `gpu_mem_gb` is under about **75**

   **Send Ishaan these five lines.**
2. `=== seed_0`, then a line every 10 steps. Over the run:
   - `loss_private` should **fall clearly**, as the model learns the planted documents.
   - `loss_dolma` should stay **roughly flat**.
3. `=== seed_1`, `=== seed_2`, and finally `=== all three seeds trained`.

## If something goes wrong
| You see | Do this |
|---|---|
| The pod restarted, or the run died | Run step 4 again. If the pod was wiped (no `.venv` folder), do step 2 first. Finished seeds are skipped, and the unfinished one continues from its last save (every 20 steps). |
| `rebuilt file differs from …` | Stop. Don't delete anything. Message Ishaan: the training data doesn't match the committed version. |
| `settings changed since …` | Stop and message Ishaan. A run is being resumed with different options. |
| `CUDA out of memory` | Stop and message Ishaan with the last 30 lines of the log. |
| `WARNING: upload of seed_N … failed` | Nothing to do now; training continues. Upload by hand in step 6. |

## 6. When it's done: send the results back
1. **Small records** (logs, settings, training order). The big weight files are excluded automatically.
   ```
   git add extension/models
   git commit -m "Extension: seed_0-2 trained"
   git push
   ```
   If `git push` asks for a password, use a GitHub personal access token, or message Ishaan to sort out another way to get the files over.
2. **Trained models (the weights):**
   - **If you did step 3:** check that https://huggingface.co/<your-hf-username>/olmo-pano-adapters has folders `seed_0`, `seed_1` and `seed_2`. Re-upload any that are missing:
     ```
     .venv/bin/hf upload $HF_REPO extension/models/seed_0 seed_0 --private --exclude "checkpoint-*"
     ```
     Swap `seed_0` for `seed_1` or `seed_2` as needed.
   - **If you didn't:** do step 3's login now, then run that upload line for all three seeds.

**Don't stop or terminate the pod until the three `seed_*` folders are on HuggingFace.**
