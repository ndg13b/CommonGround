# Changelog

## 0.2.0 (2026-10-09)

Changes from a design review of 0.1. [docs/design.md](docs/design.md) explains each one.

### Added

- **Questions**, kept apart from doubts: `ask()` raises a question, optionally about specific claims, and `answer()` offers a claim as the answer. A question counts as answered once that claim is accepted. `answers_to()` lists accepted answers.
- **`refine()`** adds a narrower version of a claim that is true but too broad, and keeps the original active. `revise()` is still the way to replace a claim that was wrong.
- **Qualifiers** on claims (`qualifiers={"for customer": "Acme Corp"}`), shown wherever the claim is.
- **Registered sources**: `add_source()`. Claims and responses can only cite registered sources, and the log records who supplied each one.
- **Participant identity**: `add_participant(..., identity=...)`. Participants sharing an identity count as one voice when deciding whether a claim is accepted.
- **Relation vocabulary**: `CommonGround(vocabulary=[...])` fixes the allowed relations; `relations_in_use()` lists the ones already used.
- **`review_queue(focus=..., limit=...)`** ranks unsettled claims by how many of the questions being asked depend on them, then by how contested they are.
- Responses can cite sources: `respond(..., sources=[...])`.
- `docs/design.md`: design rationale, known limits, an evaluation plan and related work.

### Changed

- Models must give a reason when they agree. Set `CommonGround(agreement_reason_required=...)` to change which kinds of participant this applies to.
- `open_questions()` now returns unanswered `Question`s. Its old behaviour (claims needing discussion, most contested first) is `review_queue()` with no focus.
- `find_paths()` puts the best-supported chain first among chains of equal length.
- `explain_path()`, `why()` and `to_context()` show questions, answers, refinements and qualifiers. `to_context()` also lists citable sources and the relations in use.
- `explain_path()` says "Every link is accepted" instead of "agreed by all participants", which overstated what `accepted` means.
- Log entries name their subject in `target` instead of `claim`, since they can now refer to questions and sources.
- Saved graphs include settings, sources and questions. Files saved by 0.1 cannot be loaded.
- The README describes the graph as the common ground that conversation builds, rather than as the conversation itself.

## 0.1.0 (2026-09-27)

First prototype.
