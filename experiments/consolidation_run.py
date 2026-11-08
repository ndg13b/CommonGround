"""A small experiment: train what the graph marks ready into a model, and check it with probes.

This is the first experiment from docs/learning.md, scaled down to run on a CPU.
It uses the story from examples/learning_loop.py and a small open model with a
LoRA (low-rank adaptation) adapter: a few million trainable weights added on
top of a model whose own weights stay frozen.

1. Before the news report is corrected, train an adapter on the claims the
   plan marks ready, and probe the model before and after.
2. After the correction, try two ways to update the model:
   - rebuild: train a fresh adapter from the corrected graph (the graph is the
     source of truth, the weights a rebuildable copy);
   - patch: keep training the first adapter on just what changed (editing in place).
3. Gate each update against the model currently in use. It must do what it
   is for: every fact and ripple probe passes, and every negative probe is
   rejected. And it must do no damage: no core probe that passed before now
   fails, and the model does no worse on control probes about things outside
   the graph. A rejected update leaves the model in use unchanged.

Training only on true statements tends to make a model credulous: it starts calling
everything true. By default, half of each training batch is therefore "replay":
true-or-false questions about general knowledge, half of them false, answered
correctly, in the same format the model is asked in. Rehearsing the old task
alongside the new knowledge is the usual way to keep old behaviour. None of the
replay statements are controls, and the graph's own claims are never trained in
question form, so the probes still test something the model wasn't trained on.
--no-replay turns replay off, to see the difference.

Training text is the claims written as sentences with their rationales, and
corrections for what was revised away. The probes ask true-or-false questions,
a different format from the training text, so passing them means the trained
knowledge carried over to a question it wasn't trained on.

Small models lean towards answering "True", so answers are calibrated: the
cut-off between true and false is set once, from the untrained model's answers
to the control probes, and kept for every later evaluation. A model that
becomes more credulous after training therefore shows up as failing probes.

Needs: pip install -e ".[experiments]"   (torch, transformers, peft)
Run from the repo root:  python experiments/consolidation_run.py
On a GPU, with a larger model:  python experiments/consolidation_run.py \\
    --model Qwen/Qwen2.5-1.5B-Instruct --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from commonground import CommonGround, Probe, plan_consolidation  # noqa: E402
from learning_loop import build, confirm_recall, correct_report, recheck_branch  # noqa: E402

# Statements about the world outside the graph. Training shouldn't change how
# the model judges these; if it gets worse on them, the update damaged it.
CONTROLS = [
    ("Paris is the capital of France.", True),
    ("Water boils at 100 degrees Celsius at sea level.", True),
    ("The Earth orbits the Sun.", True),
    ("A week has seven days.", True),
    ("Spiders have eight legs.", True),
    ("The Pacific is the largest ocean on Earth.", True),
    ("Humans need oxygen to survive.", True),
    ("A triangle has three sides.", True),
    ("Ice floats on water.", True),
    ("Tokyo is in Japan.", True),
    ("Cats are a kind of bird.", False),
    ("The Moon is larger than the Earth.", False),
    ("Ice is hotter than steam.", False),
    ("Two plus two equals five.", False),
    ("The Sun rises in the west.", False),
    ("Fish breathe with lungs.", False),
    ("A year has 400 days.", False),
    ("Mount Everest is in Brazil.", False),
    ("Snow is usually black.", False),
    ("Spiders have six legs.", False),
]


# General-knowledge questions rehearsed during training ("replay"), half of
# them false. None overlaps the controls, which must stay unseen.
REPLAY = [
    ("Berlin is the capital of Germany.", True),
    ("A square has four equal sides.", True),
    ("Bees make honey.", True),
    ("The Atlantic Ocean lies between Europe and America.", True),
    ("Plants need sunlight to grow.", True),
    ("A dozen means twelve.", True),
    ("Gold is a metal.", True),
    ("The heart pumps blood around the body.", True),
    ("Penguins live at the North Pole.", False),
    ("Three is an even number.", False),
    ("Copper is a gas at room temperature.", False),
    ("Rome is the capital of Spain.", False),
    ("Whales are a kind of fish.", False),
    ("A minute has one hundred seconds.", False),
    ("Lightning is colder than ice.", False),
    ("Horses have six legs.", False),
]


# ------------------------------------------------------------------ the graph

def sentence(text: str) -> str:
    return text if text.endswith(".") else text + "."


def training_texts(graph: CommonGround, plan, only: set[str] | None = None) -> list[str]:
    """Sentences to train on: each ready claim, alone and with its reason, plus corrections."""
    statements = {p.claim: p.statement for p in plan.probes}
    texts = []
    for claim in plan.ready:
        if only is not None and claim.id not in only:
            continue
        said = statements.get(claim.id) or sentence(claim.triple())
        texts += [said, f"{said} {sentence(claim.rationale)}"]
    for probe in plan.probes:
        if probe.kind == "negative" and (only is None or probe.claim in only):
            current = graph.current(probe.claim)
            replacement = statements.get(current.id, sentence(current.triple()))
            texts += [f"This is false: {probe.statement}", f"Correction: {replacement}"]
    return texts


# ---------------------------------------------------------------------- model

def load(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    return tokenizer, model


def answer_ids(tokenizer) -> tuple[int, int]:
    true_id = tokenizer.encode("True", add_special_tokens=False)[0]
    false_id = tokenizer.encode("False", add_special_tokens=False)[0]
    return true_id, false_id


def question(statement: str) -> list[dict]:
    return [{"role": "user", "content": "Is the following statement true or false?\n\n"
             f"{statement}\n\nAnswer with exactly one word, True or False."}]


def replay_texts(tokenizer) -> list[str]:
    """The replay questions with their right answers, as chat transcripts."""
    return [
        tokenizer.apply_chat_template(
            question(s) + [{"role": "assistant", "content": "True" if truth else "False"}],
            tokenize=False,
        )
        for s, truth in REPLAY
    ]


@torch.no_grad()
def p_true(model, tokenizer, statement: str, device: str) -> float:
    """How strongly the model answers True rather than False to a true-or-false question."""
    prompt = tokenizer.apply_chat_template(question(statement), tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits[0, -1]
    true_id, false_id = answer_ids(tokenizer)
    return torch.softmax(logits[[true_id, false_id]], dim=0)[0].item()


def calibrate(model, tokenizer, device: str) -> float:
    """The cut-off between true and false: midway between the model's average
    score for the true controls and for the false ones."""
    model.eval()
    scores = {True: [], False: []}
    for statement, expected in CONTROLS:
        scores[expected].append(p_true(model, tokenizer, statement, device))
    return round((sum(scores[True]) / len(scores[True]) + sum(scores[False]) / len(scores[False])) / 2, 4)


def evaluate(model, tokenizer, probes: list[Probe], device: str, cutoff: float) -> list[dict]:
    model.eval()
    rows = [{"kind": p.kind, "claim": p.claim, "statement": p.statement, "expected": p.expected}
            for p in probes]
    rows += [{"kind": "control", "claim": None, "statement": s, "expected": e} for s, e in CONTROLS]
    for row in rows:
        row["p_true"] = round(p_true(model, tokenizer, row["statement"], device), 4)
        row["passed"] = (row["p_true"] > cutoff) == row["expected"]
    return rows


def train(model, tokenizer, texts: list[str], steps: int, lr: float, batch: int,
          device: str, seed: int, replay: bool) -> list[float]:
    """Plain next-token training of the adapter on the texts, half of each batch replay if asked."""
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        own = batch // 2 if replay else batch
        chunk = rng.sample(texts, k=min(own, len(texts)))
        if replay:
            chunk += rng.sample(replay_texts(tokenizer), k=batch - own)
        enc = tokenizer([t + tokenizer.eos_token for t in chunk], return_tensors="pt",
                        padding=True).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(round(loss.item(), 4))
    return losses


def with_adapter(base, rank: int):
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(copy.deepcopy(base), config)


# ----------------------------------------------------------------------- gate

def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summary(rows: list[dict]) -> dict:
    out = {}
    for kind in ("core", "fact", "ripple", "negative", "control"):
        picked = [r for r in rows if r["kind"] == kind]
        if picked:
            out[kind] = {"passed": sum(r["passed"] for r in picked), "of": len(picked),
                         "mean_score_true": mean([r["p_true"] for r in picked if r["expected"]]),
                         "mean_score_false": mean([r["p_true"] for r in picked if not r["expected"]])}
    return out


def gate(before: list[dict], after: list[dict]) -> tuple[bool, list[str]]:
    """Accept an update only if it does what it is for and does no damage."""
    reasons = []
    for kind, failing in (("fact", "still judged false"), ("ripple", "still judged false"),
                          ("negative", "still believed")):
        missed = [r for r in after if r["kind"] == kind and not r["passed"]]
        if missed:
            total = sum(r["kind"] == kind for r in after)
            reasons.append(f"{len(missed)} of {total} {kind} probes {failing}")
    passed_before = {r["statement"] for r in before if r["kind"] == "core" and r["passed"]}
    broken = [r["statement"] for r in after
              if r["kind"] == "core" and not r["passed"] and r["statement"] in passed_before]
    reasons += [f"core probe broken: {s}" for s in broken]
    controls_before = sum(r["passed"] for r in before if r["kind"] == "control")
    controls_after = sum(r["passed"] for r in after if r["kind"] == "control")
    if controls_after < controls_before:
        reasons.append(f"control probes fell from {controls_before} to {controls_after} passed")
    return not reasons, reasons


# ------------------------------------------------------------------------ run

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--patch-steps", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-replay", dest="replay", action="store_false",
                        help="train on the graph's sentences alone, without replay")
    parser.add_argument("--out", default=str(ROOT / "experiments" / "results" / "latest.json"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    started = time.time()

    graph = build()
    confirm_recall(graph)
    plan_a = plan_consolidation(graph)
    corrected = correct_report(graph)
    recheck_branch(graph, corrected)
    plan_b = plan_consolidation(graph, changed=[corrected.id])
    changed = {corrected.id} | {c.id for c in graph.dependents(corrected.id)}
    changed |= {p.claim for p in plan_b.probes if p.kind == "negative"}

    tokenizer, base = load(args.model, args.device)
    for p in base.parameters():
        p.requires_grad_(False)
    cutoff = calibrate(base, tokenizer, args.device)
    print(f"Cut-off between true and false, from the untrained model: {cutoff}")
    results: dict = {"model": args.model, "settings": vars(args), "cutoff": cutoff, "stages": {}}

    def record(name: str, before, after, texts, losses, note: str = "") -> bool:
        accepted, reasons = gate(before, after)
        results["stages"][name] = {
            "note": note,
            "training_texts": texts, "loss_first": losses[0], "loss_last": losses[-1],
            "before": summary(before), "after": summary(after),
            "accepted": accepted, "rejected_because": reasons, "probes_after": after,
            "probes_before": before,
        }
        verdict = "accepted" if accepted else "REJECTED: " + "; ".join(reasons)
        print(f"\n== {name}: {verdict}" + (f"\n  ({note})" if note else ""))
        for kind, s in summary(after).items():
            b = summary(before).get(kind, {})
            line = f"  {kind:<8} passed {b.get('passed')}/{b.get('of')} -> {s['passed']}/{s['of']}"
            for side, label in (("mean_score_true", "true"), ("mean_score_false", "false")):
                if s[side] is not None:
                    line += f"   score of {label} statements {b.get(side)} -> {s[side]}"
            print(line)
        return accepted

    # 1. Before the correction: train what's ready. The model in use is the base model.
    in_use, in_use_name = base, "the base model"
    texts_a = training_texts(graph, plan_a)
    before_a = evaluate(in_use, tokenizer, plan_a.probes, args.device, cutoff)
    model_a = with_adapter(base, args.rank)
    losses = train(model_a, tokenizer, texts_a, args.steps, args.lr, args.batch, args.device,
                   args.seed, args.replay)
    after_a = evaluate(model_a, tokenizer, plan_a.probes, args.device, cutoff)
    accepted_a = record("1-before-correction", before_a, after_a, texts_a, losses,
                        "compared with the base model")
    if accepted_a:
        in_use, in_use_name = model_a, "update 1"

    # 2a. Rebuild from the corrected graph: a fresh adapter on the base model.
    texts_b = training_texts(graph, plan_b)
    before_b = evaluate(in_use, tokenizer, plan_b.probes, args.device, cutoff)
    model_rebuilt = with_adapter(base, args.rank)
    losses = train(model_rebuilt, tokenizer, texts_b, args.steps, args.lr, args.batch,
                   args.device, args.seed, args.replay)
    after_rebuilt = evaluate(model_rebuilt, tokenizer, plan_b.probes, args.device, cutoff)
    record("2a-rebuild-from-corrected-graph", before_b, after_rebuilt, texts_b, losses,
           f"compared with the model in use, {in_use_name}")

    # 2b. Patch update 1's adapter with only what changed.
    texts_patch = training_texts(graph, plan_b, only=changed)
    losses = train(model_a, tokenizer, texts_patch, args.patch_steps, args.lr, args.batch,
                   args.device, args.seed, args.replay)
    after_patched = evaluate(model_a, tokenizer, plan_b.probes, args.device, cutoff)
    note = f"compared with the model in use, {in_use_name}"
    if not accepted_a:
        note += "; this patches update 1, which the gate rejected"
    record("2b-patch-in-place", before_b, after_patched, texts_patch, losses, note)

    results["seconds"] = round(time.time() - started)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved results to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
          f" ({results['seconds']} s)")


if __name__ == "__main__":
    main()
