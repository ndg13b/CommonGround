"""Core data model for a shared human-model knowledge graph.

Every edge in the graph is a *claim*: someone proposed it and gave a reason.
Participants (humans or models) record a stance on each claim. Nothing is
overwritten: a revision creates a new claim that supersedes the old one, and
every action is written to an append-only log. The graph is therefore both a
body of knowledge and a readable record of how the participants reached it.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional


class Stance(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNSURE = "unsure"


class Status(str, Enum):
    PROPOSED = "proposed"      # only the proposer has endorsed it so far
    ACCEPTED = "accepted"      # someone besides the proposer agrees; nobody objects
    UNCERTAIN = "uncertain"    # someone is unsure; nobody disagrees
    DISPUTED = "disputed"      # at least one participant disagrees
    SUPERSEDED = "superseded"  # replaced by a revised claim


ACTIVE_STATUSES = frozenset(
    {Status.PROPOSED, Status.ACCEPTED, Status.UNCERTAIN, Status.DISPUTED}
)

# Lower is stronger. Used to find the weakest link in a chain of claims.
_STRENGTH = {
    Status.ACCEPTED: 0,
    Status.PROPOSED: 1,
    Status.UNCERTAIN: 2,
    Status.DISPUTED: 3,
    Status.SUPERSEDED: 4,
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Participant:
    name: str
    kind: str  # "human" or "model"


@dataclass
class Position:
    stance: Stance
    reason: str
    at: str


@dataclass
class Claim:
    id: str
    subject: str
    relation: str
    object: str
    proposed_by: str
    rationale: str
    sources: list[str]
    created_at: str
    positions: dict[str, Position] = field(default_factory=dict)
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None

    @property
    def status(self) -> Status:
        if self.superseded_by:
            return Status.SUPERSEDED
        stances = [p.stance for p in self.positions.values()]
        if Stance.DISAGREE in stances:
            return Status.DISPUTED
        if Stance.UNSURE in stances:
            return Status.UNCERTAIN
        others_agree = any(
            name != self.proposed_by and p.stance is Stance.AGREE
            for name, p in self.positions.items()
        )
        return Status.ACCEPTED if others_agree else Status.PROPOSED

    def triple(self) -> str:
        return f"{self.subject} —{self.relation}→ {self.object}"


class CommonGround:
    """A knowledge graph that humans and models build and argue about together."""

    def __init__(self, clock: Callable[[], str] = _utcnow) -> None:
        self.participants: dict[str, Participant] = {}
        self.claims: dict[str, Claim] = {}
        self.log: list[dict] = []
        self._clock = clock

    # ------------------------------------------------------------------ setup

    def add_participant(self, name: str, kind: str) -> Participant:
        if kind not in ("human", "model"):
            raise ValueError("kind must be 'human' or 'model'")
        participant = Participant(name, kind)
        self.participants[name] = participant
        return participant

    # ---------------------------------------------------------------- actions

    def propose(
        self,
        by: str,
        subject: str,
        relation: str,
        obj: str,
        rationale: str,
        sources: Iterable[str] = (),
    ) -> Claim:
        """Add a claim. A rationale is required: every link must say why it exists."""
        self._require_participant(by)
        if not rationale.strip():
            raise ValueError("A claim needs a rationale: say why you believe it.")
        now = self._clock()
        claim = Claim(
            id=f"c{len(self.claims) + 1}",
            subject=subject,
            relation=relation,
            object=obj,
            proposed_by=by,
            rationale=rationale,
            sources=list(sources),
            created_at=now,
            positions={by: Position(Stance.AGREE, "proposer", now)},
        )
        self.claims[claim.id] = claim
        self._record(by, "propose", claim.id, claim.triple())
        return claim

    def respond(self, claim_id: str, by: str, stance: str | Stance, reason: str = "") -> Claim:
        """Agree, disagree or say you're unsure. Anything but agreement needs a reason."""
        self._require_participant(by)
        claim = self._get(claim_id)
        if claim.status is Status.SUPERSEDED:
            raise ValueError(
                f"{claim_id} was superseded by {claim.superseded_by}; respond to that instead."
            )
        stance = Stance(stance)
        if stance is not Stance.AGREE and not reason.strip():
            raise ValueError(
                "Disagreeing or being unsure needs a reason, so the other side can respond to it."
            )
        claim.positions[by] = Position(stance, reason, self._clock())
        self._record(by, stance.value, claim_id, reason)
        return claim

    def revise(
        self,
        claim_id: str,
        by: str,
        reason: str,
        *,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        obj: Optional[str] = None,
        rationale: Optional[str] = None,
        sources: Optional[Iterable[str]] = None,
    ) -> Claim:
        """Replace a claim with an amended one. The original is kept and marked superseded."""
        self._require_participant(by)
        old = self._get(claim_id)
        if old.status is Status.SUPERSEDED:
            raise ValueError(f"{claim_id} was already superseded by {old.superseded_by}.")
        if not reason.strip():
            raise ValueError("A revision needs a reason.")
        new = self.propose(
            by,
            subject or old.subject,
            relation or old.relation,
            obj or old.object,
            rationale or old.rationale,
            old.sources if sources is None else sources,
        )
        new.supersedes = old.id
        old.superseded_by = new.id
        self._record(by, "revise", old.id, f"-> {new.id}: {reason}")
        return new

    # ---------------------------------------------------------------- queries

    def active_claims(self, statuses: Iterable[Status] = ACTIVE_STATUSES) -> list[Claim]:
        statuses = frozenset(statuses)
        return [c for c in self.claims.values() if c.status in statuses]

    def open_questions(self) -> list[Claim]:
        """Claims that still need discussion: not yet confirmed, uncertain or disputed."""
        pending = {Status.PROPOSED, Status.UNCERTAIN, Status.DISPUTED}
        return sorted(
            (c for c in self.claims.values() if c.status in pending),
            key=lambda c: -_STRENGTH[c.status],
        )

    def find_paths(
        self,
        start: str,
        end: str,
        max_hops: int = 4,
        statuses: Iterable[Status] = ACTIVE_STATUSES,
    ) -> list[list[Claim]]:
        """All chains of claims from `start` to `end`, shortest first (case-insensitive)."""
        key = str.casefold
        edges: dict[str, list[Claim]] = {}
        for claim in self.active_claims(statuses):
            edges.setdefault(key(claim.subject), []).append(claim)

        results: list[list[Claim]] = []
        queue: deque[tuple[str, list[Claim]]] = deque([(key(start), [])])
        while queue:
            node, path = queue.popleft()
            if path and node == key(end):
                results.append(path)
                continue
            if len(path) >= max_hops:
                continue
            visited = {key(start)} | {key(c.object) for c in path}
            for claim in edges.get(node, []):
                if key(claim.object) not in visited:
                    queue.append((key(claim.object), path + [claim]))
        return results

    def explain_path(self, path: list[Claim]) -> str:
        """Readable account of a chain, including objections and its weakest link."""
        lines = []
        for claim in path:
            src = ", ".join(claim.sources) if claim.sources else "no source: inferred"
            lines.append(f"  {claim.id}: {claim.triple()}  [{claim.status.value}; {src}]")
            for name, pos in claim.positions.items():
                if pos.stance is not Stance.AGREE:
                    lines.append(f"      {name} is {pos.stance.value}: {pos.reason}")
        weakest = max(path, key=lambda c: _STRENGTH[c.status])
        if weakest.status is Status.ACCEPTED:
            lines.append("  Every link is agreed by all participants.")
        else:
            lines.append(
                f"  Weakest link: {weakest.id} ({weakest.status.value}). "
                "The conclusion is only as firm as this."
            )
        return "\n".join(lines)

    def why(self, claim_id: str) -> str:
        """Everything recorded about a claim: who proposed it, why, reactions and history."""
        claim = self._get(claim_id)
        lines = [
            f"{claim.id}: {claim.triple()}  [{claim.status.value}]",
            f"  Proposed by {claim.proposed_by}: {claim.rationale}",
            f"  Sources: {', '.join(claim.sources) or 'none (inferred)'}",
        ]
        for name, pos in claim.positions.items():
            if name == claim.proposed_by and pos.reason == "proposer":
                continue
            lines.append(f"  {name}: {pos.stance.value}" + (f" ({pos.reason})" if pos.reason else ""))
        if claim.superseded_by:
            lines.append(f"  Superseded by {claim.superseded_by}.")
        previous = claim.supersedes
        while previous:
            old = self.claims[previous]
            lines.append(f"  Revises {old.id}: {old.triple()}")
            previous = old.supersedes
        return "\n".join(lines)

    # ----------------------------------------------------------------- export

    def to_context(self) -> str:
        """Plain-text rendering of the current graph, for pasting into any LLM prompt."""
        lines = ["Shared knowledge graph (each line: id, claim, status, sources, objections):"]
        for claim in self.active_claims():
            src = ", ".join(claim.sources) or "inferred"
            objections = [
                f"{n} {p.stance.value}: {p.reason}"
                for n, p in claim.positions.items()
                if p.stance is not Stance.AGREE
            ]
            extra = f" | {'; '.join(objections)}" if objections else ""
            lines.append(f"- {claim.id}: {claim.triple()} | {claim.status.value} | {src}{extra}")
        return "\n".join(lines)

    def to_mermaid(self, statuses: Iterable[Status] = ACTIVE_STATUSES) -> str:
        """Mermaid diagram: solid arrows are agreed, dotted arrows are still open."""
        node_ids: dict[str, str] = {}

        def node(name: str) -> str:
            return node_ids.setdefault(name, f"n{len(node_ids)}")

        edges = []
        for claim in self.active_claims(statuses):
            a, b = node(claim.subject), node(claim.object)
            label = claim.relation.replace('"', "'")
            if claim.status is Status.ACCEPTED:
                edges.append(f'    {a} -- "{label}" --> {b}')
            else:
                edges.append(f'    {a} -. "{label} ({claim.status.value})" .-> {b}')
        nodes = [f'    {i}["{name.replace(chr(34), chr(39))}"]' for name, i in node_ids.items()]
        return "\n".join(["graph LR", *nodes, *edges])

    def to_dict(self) -> dict:
        def claim_dict(c: Claim) -> dict:
            return {
                "id": c.id,
                "subject": c.subject,
                "relation": c.relation,
                "object": c.object,
                "proposed_by": c.proposed_by,
                "rationale": c.rationale,
                "sources": c.sources,
                "created_at": c.created_at,
                "positions": {
                    n: {"stance": p.stance.value, "reason": p.reason, "at": p.at}
                    for n, p in c.positions.items()
                },
                "supersedes": c.supersedes,
                "superseded_by": c.superseded_by,
            }

        return {
            "participants": [{"name": p.name, "kind": p.kind} for p in self.participants.values()],
            "claims": [claim_dict(c) for c in self.claims.values()],
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommonGround":
        graph = cls()
        for p in data["participants"]:
            graph.add_participant(p["name"], p["kind"])
        for c in data["claims"]:
            positions = {
                n: Position(Stance(p["stance"]), p["reason"], p["at"])
                for n, p in c["positions"].items()
            }
            graph.claims[c["id"]] = Claim(**{**c, "positions": positions})
        graph.log = list(data["log"])
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CommonGround":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---------------------------------------------------------------- helpers

    def _require_participant(self, name: str) -> None:
        if name not in self.participants:
            raise KeyError(f"Unknown participant: {name!r}")

    def _get(self, claim_id: str) -> Claim:
        try:
            return self.claims[claim_id]
        except KeyError:
            raise KeyError(f"Unknown claim: {claim_id!r}") from None

    def _record(self, actor: str, action: str, claim_id: str, detail: str = "") -> None:
        self.log.append(
            {"at": self._clock(), "actor": actor, "action": action, "claim": claim_id, "detail": detail}
        )
