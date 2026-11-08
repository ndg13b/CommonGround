# Experiments

## Training what the graph marks ready into a small model

[`consolidation_run.py`](consolidation_run.py) is the first experiment from [docs/learning.md](../docs/learning.md#a-first-experiment), scaled down to run on a CPU. It asks: if a model is trained on the claims the graph marks ready, does it come to believe them, does it drop what was revised away, and does the gate catch updates that do damage?

### What it does

It uses the story from [`examples/learning_loop.py`](../examples/learning_loop.py) and a small open model with a LoRA (low-rank adaptation) adapter: a small set of extra weights trained on top of the model, whose own weights stay frozen.

1. **Before the news report is corrected:** train an adapter on the claims the plan marks ready, and put the probes to the model before and after.
2. **After the correction,** try two ways to update the model:
   - **rebuild:** train a fresh adapter from the corrected graph. This treats the graph as the source of truth and the weights as a rebuildable copy.
   - **patch:** keep training the first adapter on just what changed, which is editing in place.
3. **Gate each update** against the model in use. It must do what it is for: every fact and ripple probe passes, and every negative probe (what was revised away) is rejected. And it must do no damage: no core probe that passed before now fails, and the model does no worse on 20 *control* probes, true/false statements about things outside the graph ("Paris is the capital of France"). A rejected update leaves the model in use unchanged.

The training text is the claims written as sentences, with their rationales, plus corrections for what was revised away. The probes ask true-or-false questions, a different format from the training text, so a probe passes only if the trained knowledge carried over to a question the model wasn't trained on.

A probe's score is how strongly the model answers "True" rather than "False", from 0 to 1. Small models lean towards "True", so the cut-off is calibrated: it is set once, midway between the untrained model's average scores for the true and the false controls, and kept for every later evaluation. A model that becomes more credulous after training therefore fails probes, rather than having the cut-off move with it.

Training only on true statements tends to make a model credulous. So by default half of each training batch is *replay*: true/false questions about general knowledge, half of them false, answered correctly, in the same format the model is asked in. None of the replay statements are controls, and the graph's own claims are never trained in question form, so the probes still test something the model wasn't trained on. `--no-replay` turns replay off.

### Running it

```bash
pip install -e ".[experiments]"          # torch, transformers, peft
python experiments/consolidation_run.py  # 2 to 4 minutes on a 4-core CPU
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct` (about 500 million parameters, Apache 2.0 licence). Results go to `experiments/results/latest.json`, with every probe's score before and after each stage.

Why not something smaller: SmolLM2-135M-Instruct and SmolLM2-360M-Instruct were tried first, and neither can tell true statements from false ones (an AUROC of 0.50–0.67 on the controls, where 0.5 is chance). Qwen2.5-0.5B-Instruct separates them well (0.96), though it leans heavily towards answering "True", hence the calibration.

### On a cloud GPU later

The script takes any chat model from Hugging Face. On a machine with a GPU:

```bash
python experiments/consolidation_run.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --out experiments/results/qwen-1.5b.json
```

Larger models answer true/false questions far more reliably, so the probes become a much sharper measurement. `--steps`, `--lr` and `--rank` control the training.

### Results

Four settings, each with three updates: with and without replay, at learning rates of 1e-3 and 2e-4, all with seed 0, 30 training steps (15 for the patch), a LoRA rank of 16 and batches of 8. Before training, the model passes 1 of 5 fact probes, 1 of 2 core probes and 16 of 20 controls. The cells show how many probes pass *after* each update.

| Setting | Stage | Facts | Ripples | Revised away rejected | Core | Controls | Gate |
|---|---|---|---|---|---|---|---|
| No replay, lr 1e-3 | 1. Train what's ready | 1/5 | – | – | 2/2 | 15/20 | rejected |
| No replay, lr 1e-3 | 2a. Rebuild | 3/3 | 2/2 | 0/2 | 2/2 | 11/20 | rejected |
| No replay, lr 1e-3 | 2b. Patch | 0/3 | 0/2 | 2/2 | 0/2 | 10/20 | rejected |
| No replay, lr 2e-4 | 1. Train what's ready | 5/5 | – | – | 2/2 | 10/20 | rejected |
| No replay, lr 2e-4 | 2a. Rebuild | 3/3 | 2/2 | 0/2 | 2/2 | 11/20 | rejected |
| No replay, lr 2e-4 | 2b. Patch | 1/3 | 0/2 | 2/2 | 2/2 | 13/20 | rejected |
| Replay, lr 1e-3 | 1. Train what's ready | 2/5 | – | – | 2/2 | 16/20 | rejected |
| Replay, lr 1e-3 | 2a. Rebuild | 0/3 | 0/2 | 2/2 | 0/2 | 10/20 | rejected |
| Replay, lr 1e-3 | 2b. Patch | 0/3 | 0/2 | 2/2 | 2/2 | 15/20 | rejected |
| Replay, lr 2e-4 | 1. Train what's ready | 0/5 | – | – | 0/2 | 14/20 | rejected |
| Replay, lr 2e-4 | 2a. Rebuild | 0/3 | 0/2 | 2/2 | 0/2 | 17/20 | rejected |
| Replay, lr 2e-4 | 2b. Patch | 0/3 | 0/2 | 2/2 | 2/2 | 16/20 | rejected |

**No update passed the gate.** At this scale, training on the graph's claims changed how the model judges *everything*, rather than adding the specific facts. It failed in two mirror-image ways:

- **Without replay, the model became credulous.** Rebuilt from the corrected graph, it learned the new facts and the claims resting on the change (3/3 and 2/2, scoring 0.84 to 0.94). But it still believed both claims that had been revised away, and it called false control statements true (their average score rose from 0.61 to between 0.84 and 0.92). It hadn't learned "Porto, not Lyon"; it had learned to say "True".
- **With replay, the model became sceptical.** It kept judging general knowledge (14 to 17 of 20 controls) but rejected the graph's new facts (their average score fell from 0.55 to between 0.02 and 0.41). Rejecting both revised-away claims is no achievement here, since it rejected everything new.

The gate caught every one of these. It also exposed a flaw in its own first version, which only checked for damage: that version accepted a patch that had left the model strongly *disbelieving* the facts it was meant to learn. The gate now also requires an update to do what it is for. Similarly, a first version of replay used plain sentences ("This is false: …"), which didn't stop the model becoming credulous; rehearsing true/false questions did, but overshot into scepticism.

Rebuild against patch has no clear answer at this scale. Without replay, rebuilding learned the corrected facts and ripples where patching didn't, but neither passed.

What this shows: the pipeline works end to end, and the gate does its job, including catching changes to the model's behaviour that a simple accuracy figure would miss. What it doesn't yet show is a training recipe that adds knowledge without side effects, which is exactly the hard problem [docs/learning.md](../docs/learning.md#why-it-is-hard) describes.

Limits: one seed per setting; one small model whose true/false judgement is fragile; 20 controls and only nine graph probes; and true/false questions are a shallow test.

### Next

- **A larger model on a GPU.** Larger models judge true/false far more reliably, so the probes become a sharper measurement. The script needs only `--model` and `--device cuda`.
- **Several seeds per setting,** to separate real effects from noise.
- **Training that shows both sides.** The graph can generate false versions of its own claims, for example by swapping objects ("Acme Corp buys batteries from Lyon plant"), so that learning a fact doesn't amount to learning to say "True".
- **A comparison with direct editing** methods such as ROME and MEMIT, which change specific facts in the weights instead of fine-tuning.

The runs above loaded a local copy of `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae5576`, checked against its published checksum. The saved results name the model on the Hub and that revision.
