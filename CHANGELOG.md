# Changelog

## Unreleased

First steps toward letting a model learn from the graph. [docs/learning.md](docs/learning.md) explains the design.

### Added

- **Levels**: `propose(..., level=...)` places a claim among principles, concepts and facts, from most to least entrenched. Only humans may propose, revise or refine principles. `revise()` and `refine()` keep the level unless given a new one.
- **What a claim rests on**: `depends_on=[...]` on `propose()`, `revise()` and `refine()`. Revisions and refinements inherit the current versions of what the original rested on. `dependents()` lists everything downstream of a claim. A revision that would rest on its own dependents is refused, so support can't be circular.
- **`needs_recheck()`** lists active claims whose support is in doubt, because something they rest on was revised, isn't accepted, or needs re-checking itself. Most entrenched claims come first. Revising a claim with nothing changed re-affirms it.
- **Independent sources**: `add_source(..., origin=...)` records what a source repeats, and `origin_of()` finds the original.
- **Source ratings**: `add_source(..., reliability=...)` and `rate_source()` rate a source low, unrated, moderate or high. Only people, or the system that supplies a source, can rate it.
- **Sentences**: `state()` adds a claim written as a plain sentence, such as a principle. `revise()` and `refine()` take `text=`. Sentences aren't edges of the graph, so `find_paths()`, `to_mermaid()` and `relations_in_use()` leave them out.
- **`reaffirm()`** re-checks a flagged claim in place when it still holds: it keeps its id and agreement and rests on the current versions. It needs a reason and settled dependencies; models may re-check only facts, and only claims from another identity.
- **`plan_consolidation(graph, thresholds, changed)`** (new `consolidation` module) sorts active claims into those ready to be trained into weights and those held back, with reasons. The bar is a `Threshold` per level (agreeing voices, human voices, independent sources), with `DEFAULT_THRESHOLDS`. By default facts need two independent sources rated moderate or better, concepts need a human among two voices, and principles need three voices including a human. It generates `Probe`s: core (every settled principle and concept, even one held back from training), fact, negative and ripple.
- `current()` returns the latest version of a claim; `questions_on()` lists the questions about a claim.
- `examples/learning_loop.py` and `docs/learning.md`.
- `experiments/consolidation_run.py`: trains the claims a plan marks ready into a small model's LoRA adapter on a CPU, probes it before and after, and gates each update (install with `pip install -e ".[experiments]"`). Results from Qwen2.5-0.5B-Instruct are in `experiments/README.md` and `experiments/results/`: the gate rejected all 12 updates.

### Changed

- `to_context()` groups claims under Principles, Concepts and Facts when there are principles or concepts. It also shows what each claim rests on, which claims need re-checking, and which sources repeat another.
- `why()` shows a claim's level when it isn't a fact, what it rests on, what it supports, and why it needs re-checking.
- `explain_path()` reports claims in the chain that need re-checking, and `find_paths()` ranks such chains lower among chains of equal length.
- `propose()` now requires a non-empty subject, relation and object; use `state()` for a sentence.
- Saved graphs include each claim's level, dependencies, sentence and re-checks, and each source's origin and rating. Files saved by 0.2 still load.

## 0.2.0 (2026-10-09)

Changes from a design review of 0.1. [docs/design.md](docs/design.md) explains each one.

### Added

- **Questions**, kept apart from doubts: `ask()` raises a question, optionally about specific claims, and `answer()` offers a claim as the answer (recorded as an `Offer`). A question counts as answered once a participant independent of the offerer (not sharing their identity) agrees with the claim after the offer, leaving it accepted. A question can't be answered by the claim it is about, or by a copy of it. `answers_to()` lists the claims that answer a question.
- **`refine()`** adds a narrower version of a claim that is true but too broad, and keeps the original active. Questions about the original, or about later revisions of it, apply to its refinements too. `revise()` is still the way to replace a claim that was wrong.
- **Qualifiers** on claims (`qualifiers={"for customer": "Acme Corp"}`), shown wherever the claim is.
- **Registered sources**: `add_source()`. Claims and responses can only cite registered sources, and the log records who supplied each one.
- **Participant identity**: `add_participant(..., identity=...)`. Participants sharing an identity count as one voice when deciding whether a claim is accepted.
- **Relation vocabulary**: `CommonGround(vocabulary=[...])` fixes the allowed relations; `relations_in_use()` lists the ones already used.
- **`review_queue(focus=..., limit=...)`** ranks unsettled claims by how many of the questions being asked depend on them, then by how contested they are.
- Responses can cite sources: `respond(..., sources=[...])`. The log keeps them, so changing a response doesn't erase them.
- `docs/design.md`: design rationale, known limits, an evaluation plan and related work.

### Changed

- Models must give a reason when they agree. Set `CommonGround(agreement_reason_required=...)` to change which kinds of participant this applies to.
- `add_participant()` with a name that is already registered returns that participant, or raises `ValueError` if the kind or identity differ. 0.1 silently replaced it.
- `Position` takes required `identity` and `seq` arguments before `sources`. `seq` is its place in the log, so the order of agreement can be checked. `Position(stance, reason, at)` no longer works.
- `open_questions()` now returns unanswered `Question`s. Its old behaviour (claims needing discussion, most contested first) is `review_queue()` with no focus.
- `find_paths()` puts the best-supported chain first among chains of equal length.
- `explain_path()` and `why()` show qualifiers, the questions on each claim and which questions it answers; `why()` also shows refinements. `to_context()` shows qualifiers, which claim refines which, and every question with its answer, and lists citable sources and the relations in use.
- `explain_path()` says "Every link is accepted" instead of "agreed by all participants", which overstated what `accepted` means.
- Log entries name their subject in `target` instead of `claim`, since they can now refer to questions and sources.
- `to_dict()` returns an independent copy, and `from_dict()` keeps no references to the dict it is given. In 0.1, editing an export could change the graph.
- Saved graphs include settings, sources and questions. Files saved by 0.1 cannot be loaded.
- The README describes the graph as the common ground that conversation builds, rather than as the conversation itself.

## 0.1.0 (2026-09-27)

First prototype.
