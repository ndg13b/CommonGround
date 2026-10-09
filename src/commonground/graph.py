"""Core data model for a shared human-model knowledge graph.

The graph holds the *common ground* between its participants: what they have
claimed, why, and how far they agree. The conversation that builds it can
happen anywhere, usually in plain language; each turn adds to or changes the
graph, and both sides cite its claims and questions by id.

Every edge in the graph is a *claim*: someone proposed it and gave a reason.
Participants (humans or models) record a stance on each claim, and raise
*questions* separately from doubts. Nothing is overwritten: a claim that was
wrong is revised (the new claim supersedes it), and a claim that was true but
too broad is refined (both stay). Every action is written to an append-only
log, so the graph is both a body of knowledge and a readable record of how the
participants reached it.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

KINDS = ("human", "model")
SYSTEM = "system"  # actor recorded for sources the system supplies


class Stance(str, Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    UNSURE = "unsure"  # doubts the claim is true; to ask about it instead, use ask()


class Status(str, Enum):
    PROPOSED = "proposed"      # no independent participant has agreed yet
    ACCEPTED = "accepted"      # an independent participant agrees; nobody objects
    UNCERTAIN = "uncertain"    # someone doubts it; nobody disagrees
    DISPUTED = "disputed"      # at least one participant disagrees
    SUPERSEDED = "superseded"  # replaced by a revised claim


ACTIVE_STATUSES = frozenset(
    {Status.PROPOSED, Status.ACCEPTED, Status.UNCERTAIN, Status.DISPUTED}
)
PENDING_STATUSES = frozenset({Status.PROPOSED, Status.UNCERTAIN, Status.DISPUTED})

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


def _qualifier_text(qualifiers: dict[str, str]) -> str:
    if not qualifiers:
        return ""
    return " (" + ", ".join(f"{k}: {v}" for k, v in qualifiers.items()) + ")"


@dataclass
class Participant:
    name: str
    kind: str      # "human" or "model"
    identity: str  # participants that share an identity count as one voice


@dataclass
class Source:
    id: str
    description: str
    added_by: Optional[str]  # None when the system supplied it
    added_at: str


@dataclass
class Position:
    stance: Stance
    reason: str
    at: str
    identity: str  # the responder's identity, so agreement can be checked for independence
    sources: list[str] = field(default_factory=list)


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
    qualifiers: dict[str, str] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    refines: Optional[str] = None

    @property
    def status(self) -> Status:
        if self.superseded_by:
            return Status.SUPERSEDED
        stances = [p.stance for p in self.positions.values()]
        if Stance.DISAGREE in stances:
            return Status.DISPUTED
        if Stance.UNSURE in stances:
            return Status.UNCERTAIN
        # Only independent agreement counts: not from the proposer, and not from
        # anyone sharing the proposer's identity (another instance of the same model).
        proposer = self.positions.get(self.proposed_by)
        proposer_identity = proposer.identity if proposer else self.proposed_by
        others_agree = any(
            p.stance is Stance.AGREE and p.identity != proposer_identity
            for p in self.positions.values()
        )
        return Status.ACCEPTED if others_agree else Status.PROPOSED

    def triple(self) -> str:
        return f"{self.subject} —{self.relation}→ {self.object}{_qualifier_text(self.qualifiers)}"


@dataclass
class Question:
    id: str
    text: str
    asked_by: str
    created_at: str
    about: list[str] = field(default_factory=list)    # claims it concerns
    answers: list[str] = field(default_factory=list)  # claims offered as answers


class CommonGround:
    """A knowledge graph that humans and models build and argue about together.

    `vocabulary`, if given, is the closed list of relations claims may use.
    `agreement_reason_required` lists the participant kinds that must give a
    reason when they agree (models by default; see respond()).
    """

    def __init__(
        self,
        clock: Callable[[], str] = _utcnow,
        *,
        vocabulary: Optional[Iterable[str]] = None,
        agreement_reason_required: Iterable[str] = ("model",),
    ) -> None:
        required = frozenset(agreement_reason_required)
        if not required <= set(KINDS):
            raise ValueError("agreement_reason_required may only contain 'human' and 'model'")
        self.participants: dict[str, Participant] = {}
        self.sources: dict[str, Source] = {}
        self.claims: dict[str, Claim] = {}
        self.questions: dict[str, Question] = {}
        self.log: list[dict] = []
        self.vocabulary: Optional[list[str]] = None if vocabulary is None else list(vocabulary)
        self.agreement_reason_required = required
        self._clock = clock

    # ------------------------------------------------------------------ setup

    def add_participant(self, name: str, kind: str, identity: Optional[str] = None) -> Participant:
        """Register a human or model.

        Participants that share an `identity` (say, two instances of the same
        model) count as one voice when deciding whether a claim is accepted.
        It defaults to the participant's name.
        """
        if kind not in KINDS:
            raise ValueError("kind must be 'human' or 'model'")
        if name == SYSTEM:
            raise ValueError(f"{SYSTEM!r} is reserved for sources the system supplies.")
        participant = Participant(name, kind, identity or name)
        self.participants[name] = participant
        return participant

    def add_source(self, source_id: str, description: str, by: Optional[str] = None) -> Source:
        """Register a document or record that claims and responses can cite.

        Only registered sources can be cited, so all cited evidence is on record
        along with who supplied it: the system (`by=None`, for example the
        documents a model was given) or a named participant.
        """
        if by is not None:
            self._require_participant(by)
        if source_id in self.sources:
            raise ValueError(f"Source {source_id!r} is already registered.")
        if not description.strip():
            raise ValueError("A source needs a description.")
        source = Source(source_id, description, by, self._clock())
        self.sources[source_id] = source
        self._record(by or SYSTEM, "add source", source_id, description)
        return source

    # ---------------------------------------------------------------- actions

    def propose(
        self,
        by: str,
        subject: str,
        relation: str,
        obj: str,
        rationale: str,
        sources: Iterable[str] = (),
        *,
        qualifiers: Optional[dict[str, str]] = None,
    ) -> Claim:
        """Add a claim. A rationale is required: every link must say why it exists.

        `qualifiers` hold conditions such as {"for customer": "Acme"} or
        {"from": "2019", "until": "2023"}, so they don't end up in the relation.
        """
        participant = self._require_participant(by)
        if not rationale.strip():
            raise ValueError("A claim needs a rationale: say why you believe it.")
        if self.vocabulary is not None and relation not in self.vocabulary:
            raise ValueError(
                f"Unknown relation {relation!r}. Allowed: {', '.join(self.vocabulary)}."
            )
        sources = self._check_sources(sources)
        now = self._clock()
        claim = Claim(
            id=f"c{len(self.claims) + 1}",
            subject=subject,
            relation=relation,
            object=obj,
            proposed_by=by,
            rationale=rationale,
            sources=sources,
            created_at=now,
            qualifiers=dict(qualifiers or {}),
            positions={by: Position(Stance.AGREE, "proposer", now, participant.identity)},
        )
        self.claims[claim.id] = claim
        self._record(by, "propose", claim.id, claim.triple())
        return claim

    def respond(
        self,
        claim_id: str,
        by: str,
        stance: str | Stance,
        reason: str = "",
        sources: Iterable[str] = (),
    ) -> Claim:
        """Agree, disagree or say you doubt a claim.

        Disagreeing or doubting always needs a reason. So does agreeing, for the
        kinds in `agreement_reason_required` (models, by default): language
        models tend to agree too readily, so an unexplained "agree" from one
        says little. To ask about a claim without doubting it, use ask().
        """
        participant = self._require_participant(by)
        claim = self._get(claim_id)
        if claim.status is Status.SUPERSEDED:
            raise ValueError(
                f"{claim_id} was superseded by {claim.superseded_by}; respond to that instead."
            )
        stance = Stance(stance)
        if not reason.strip():
            if stance is not Stance.AGREE:
                raise ValueError(
                    "Disagreeing or being unsure needs a reason, so the other side can respond to it."
                )
            if participant.kind in self.agreement_reason_required:
                raise ValueError(
                    f"Agreement from a {participant.kind} needs a reason: say what convinced you."
                )
        sources = self._check_sources(sources)
        claim.positions[by] = Position(stance, reason, self._clock(), participant.identity, sources)
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
        qualifiers: Optional[dict[str, str]] = None,
    ) -> Claim:
        """Replace a claim that was wrong. The original is kept and marked superseded.

        If the original was true but too broad, use refine() instead.
        """
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
            qualifiers=old.qualifiers if qualifiers is None else qualifiers,
        )
        new.supersedes = old.id
        old.superseded_by = new.id
        self._record(by, "revise", old.id, f"-> {new.id}: {reason}")
        return new

    def refine(
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
        qualifiers: Optional[dict[str, str]] = None,
    ) -> Claim:
        """Add a narrower version of a claim that is true but too broad. Both stay active.

        `qualifiers` are added to the original's. If the original was wrong,
        use revise() instead.
        """
        self._require_participant(by)
        old = self._get(claim_id)
        if old.status is Status.SUPERSEDED:
            raise ValueError(f"{claim_id} was superseded by {old.superseded_by}; refine that instead.")
        if not reason.strip():
            raise ValueError("A refinement needs a reason.")
        merged = {**old.qualifiers, **(qualifiers or {})}
        triple = (subject or old.subject, relation or old.relation, obj or old.object)
        if triple == (old.subject, old.relation, old.object) and merged == old.qualifiers:
            raise ValueError(
                "A refinement must narrow the claim: change its subject, relation, object or qualifiers."
            )
        new = self.propose(
            by,
            *triple,
            rationale or old.rationale,
            old.sources if sources is None else sources,
            qualifiers=merged,
        )
        new.refines = old.id
        self._record(by, "refine", old.id, f"-> {new.id}: {reason}")
        return new

    # -------------------------------------------------------------- questions

    def ask(self, by: str, text: str, about: Iterable[str] = ()) -> Question:
        """Raise a question, optionally about specific claims.

        A question is not a doubt: asking "which plant makes Acme's cells?" about
        "VoltCo makes cells at Lyon" leaves that claim's status alone, but every
        chain through the claim reports the question until it is answered.
        """
        self._require_participant(by)
        if not text.strip():
            raise ValueError("A question needs text.")
        about = list(about)
        for claim_id in about:
            if self._get(claim_id).status is Status.SUPERSEDED:
                raise ValueError(f"{claim_id} was superseded; ask about the claim that replaced it.")
        question = Question(f"q{len(self.questions) + 1}", text, by, self._clock(), about)
        self.questions[question.id] = question
        self._record(by, "ask", question.id, text)
        return question

    def answer(self, question_id: str, by: str, claim_id: str) -> Question:
        """Offer a claim as the answer to a question.

        The question counts as answered once an offered claim is accepted, so an
        answer needs independent agreement like any other claim.
        """
        self._require_participant(by)
        question = self._get_question(question_id)
        claim = self._get(claim_id)
        if claim.status is Status.SUPERSEDED:
            raise ValueError(f"{claim_id} was superseded by {claim.superseded_by}; offer that instead.")
        if claim.id not in question.answers:
            question.answers.append(claim.id)
        self._record(by, "answer", question.id, f"<- {claim.id}")
        return question

    def answers_to(self, question_id: str) -> list[Claim]:
        """Accepted claims offered as answers to a question, following any revisions."""
        question = self._get_question(question_id)
        answers: dict[str, Claim] = {}
        for claim_id in question.answers:
            claim = self._current(claim_id)
            if claim.status is Status.ACCEPTED:
                answers[claim.id] = claim
        return list(answers.values())

    def open_questions(self) -> list[Question]:
        """Questions with no accepted answer yet, oldest first."""
        return [q for q in self.questions.values() if not self.answers_to(q.id)]

    # ---------------------------------------------------------------- queries

    def active_claims(self, statuses: Iterable[Status] = ACTIVE_STATUSES) -> list[Claim]:
        statuses = frozenset(statuses)
        return [c for c in self.claims.values() if c.status in statuses]

    def relations_in_use(self) -> list[str]:
        """Relations used by active claims, most used first, so new claims can reuse them."""
        counts = Counter(c.relation for c in self.active_claims())
        return [relation for relation, _ in counts.most_common()]

    def review_queue(
        self, focus: Iterable[tuple[str, str]] = (), limit: Optional[int] = None
    ) -> list[Claim]:
        """Claims still awaiting agreement, in the order a person should check them.

        `focus` is the questions currently being asked, as (start, end) pairs like
        find_paths takes. Claims that more of them depend on come first, then
        the most contested. With no focus, it is simply most contested first.
        """
        depended_on: Counter[str] = Counter()
        for start, end in focus:
            depended_on.update({c.id for path in self.find_paths(start, end) for c in path})
        queue = sorted(
            (c for c in self.claims.values() if c.status in PENDING_STATUSES),
            key=lambda c: (-depended_on[c.id], -_STRENGTH[c.status]),
        )
        return queue if limit is None else queue[:limit]

    def find_paths(
        self,
        start: str,
        end: str,
        max_hops: int = 4,
        statuses: Iterable[Status] = ACTIVE_STATUSES,
    ) -> list[list[Claim]]:
        """All chains of claims from `start` to `end` (case-insensitive).

        Shortest first; among chains of equal length, best supported first.
        """
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
        return sorted(
            results,
            key=lambda p: (len(p), max(_STRENGTH[c.status] for c in p), len(self._unresolved(p))),
        )

    def explain_path(self, path: list[Claim]) -> str:
        """Readable account of a chain: objections, questions and its weakest link."""
        lines = []
        for claim in path:
            src = ", ".join(claim.sources) if claim.sources else "no source: inferred"
            lines.append(f"  {claim.id}: {claim.triple()}  [{claim.status.value}; {src}]")
            for name, pos in claim.positions.items():
                if pos.stance is not Stance.AGREE:
                    lines.append(f"      {name} is {pos.stance.value}: {pos.reason}")
            for q in self._questions_on(claim):
                lines.append(f"      {q.id} from {q.asked_by} ({self._answer_note(q)}): {q.text}")
            for q in self._questions_answered_by(claim):
                lines.append(f"      answers {q.id}: {q.text}")

        weakest = max(path, key=lambda c: _STRENGTH[c.status])
        unresolved = self._unresolved(path)
        if weakest.status is not Status.ACCEPTED:
            lines.append(
                f"  Weakest link: {weakest.id} ({weakest.status.value}). "
                "The conclusion is only as firm as this."
            )
        for claim, q in unresolved:
            answers = ", ".join(c.id for c in self.answers_to(q.id))
            if answers:
                lines.append(
                    f"  {q.id} on {claim.id} was answered by {answers}, which is not in this "
                    "chain, so this chain alone does not settle it."
                )
            else:
                lines.append(
                    f"  {q.id} on {claim.id} is still open, so this chain may not establish "
                    "the conclusion."
                )
        if weakest.status is Status.ACCEPTED and not unresolved:
            lines.append("  Every link is accepted and no question about them is open.")
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
            details = [pos.reason] if pos.reason else []
            if pos.sources:
                details.append(f"sources: {', '.join(pos.sources)}")
            lines.append(f"  {name}: {pos.stance.value}" + (f" ({'; '.join(details)})" if details else ""))
        for q in self._questions_on(claim):
            lines.append(f"  {q.id} from {q.asked_by} ({self._answer_note(q)}): {q.text}")
        for q in self._questions_answered_by(claim):
            lines.append(f"  Answers {q.id}: {q.text}")
        if claim.refines:
            lines.append(f"  Refines {claim.refines}: {self.claims[claim.refines].triple()}")
        refined_by = [c.id for c in self.claims.values() if c.refines == claim.id]
        if refined_by:
            lines.append(f"  Refined by {', '.join(refined_by)}.")
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
        """Plain-text rendering of the common ground so far, for pasting into any LLM prompt."""
        lines = ["Claims (each line: id, claim, status, sources, objections):"]
        for claim in self.active_claims():
            src = ", ".join(claim.sources) or "inferred"
            objections = [
                f"{n} {p.stance.value}: {p.reason}"
                for n, p in claim.positions.items()
                if p.stance is not Stance.AGREE
            ]
            extra = f" | {'; '.join(objections)}" if objections else ""
            lines.append(f"- {claim.id}: {claim.triple()} | {claim.status.value} | {src}{extra}")
        questions = self.open_questions()
        if questions:
            lines.append("Open questions:")
            for q in questions:
                about = f", about {', '.join(q.about)}" if q.about else ""
                lines.append(f"- {q.id} ({q.asked_by}{about}): {q.text}")
        if self.sources:
            lines.append("Sources (only these can be cited):")
            for s in self.sources.values():
                who = f"added by {s.added_by}" if s.added_by else "supplied by the system"
                lines.append(f"- {s.id}: {s.description} ({who})")
        if self.vocabulary is not None:
            lines.append(f"Allowed relations: {', '.join(self.vocabulary)}")
        else:
            relations = self.relations_in_use()
            if relations:
                lines.append(
                    "Relations in use (reuse one rather than inventing a synonym): "
                    + ", ".join(relations)
                )
        return "\n".join(lines)

    def to_mermaid(self, statuses: Iterable[Status] = ACTIVE_STATUSES) -> str:
        """Mermaid diagram: solid arrows are accepted, dotted arrows are still open."""
        node_ids: dict[str, str] = {}

        def node(name: str) -> str:
            return node_ids.setdefault(name, f"n{len(node_ids)}")

        edges = []
        for claim in self.active_claims(statuses):
            a, b = node(claim.subject), node(claim.object)
            label = (claim.relation + _qualifier_text(claim.qualifiers)).replace('"', "'")
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
                "qualifiers": c.qualifiers,
                "positions": {
                    n: {
                        "stance": p.stance.value,
                        "reason": p.reason,
                        "at": p.at,
                        "identity": p.identity,
                        "sources": p.sources,
                    }
                    for n, p in c.positions.items()
                },
                "supersedes": c.supersedes,
                "superseded_by": c.superseded_by,
                "refines": c.refines,
            }

        return {
            "settings": {
                "vocabulary": self.vocabulary,
                "agreement_reason_required": sorted(self.agreement_reason_required),
            },
            "participants": [asdict(p) for p in self.participants.values()],
            "sources": [asdict(s) for s in self.sources.values()],
            "claims": [claim_dict(c) for c in self.claims.values()],
            "questions": [asdict(q) for q in self.questions.values()],
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommonGround":
        settings = data["settings"]
        graph = cls(
            vocabulary=settings["vocabulary"],
            agreement_reason_required=settings["agreement_reason_required"],
        )
        for p in data["participants"]:
            graph.add_participant(p["name"], p["kind"], p["identity"])
        for s in data["sources"]:
            graph.sources[s["id"]] = Source(**s)
        for c in data["claims"]:
            positions = {
                n: Position(Stance(p["stance"]), p["reason"], p["at"], p["identity"], p["sources"])
                for n, p in c["positions"].items()
            }
            graph.claims[c["id"]] = Claim(**{**c, "positions": positions})
        for q in data["questions"]:
            graph.questions[q["id"]] = Question(**q)
        graph.log = list(data["log"])
        return graph

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CommonGround":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---------------------------------------------------------------- helpers

    def _require_participant(self, name: str) -> Participant:
        try:
            return self.participants[name]
        except KeyError:
            raise KeyError(f"Unknown participant: {name!r}") from None

    def _check_sources(self, sources: Iterable[str]) -> list[str]:
        sources = list(sources)
        unknown = [s for s in sources if s not in self.sources]
        if unknown:
            raise KeyError(
                f"Unknown source(s): {', '.join(map(repr, unknown))}. Register them with add_source()."
            )
        return sources

    def _get(self, claim_id: str) -> Claim:
        try:
            return self.claims[claim_id]
        except KeyError:
            raise KeyError(f"Unknown claim: {claim_id!r}") from None

    def _get_question(self, question_id: str) -> Question:
        try:
            return self.questions[question_id]
        except KeyError:
            raise KeyError(f"Unknown question: {question_id!r}") from None

    def _current(self, claim_id: str) -> Claim:
        """The latest revision of a claim."""
        claim = self.claims[claim_id]
        while claim.superseded_by:
            claim = self.claims[claim.superseded_by]
        return claim

    def _questions_on(self, claim: Claim) -> list[Question]:
        """Questions asked about this claim or any earlier claim it revised."""
        lineage = set()
        current: Optional[Claim] = claim
        while current is not None:
            lineage.add(current.id)
            current = self.claims.get(current.supersedes) if current.supersedes else None
        return [q for q in self.questions.values() if lineage.intersection(q.about)]

    def _questions_answered_by(self, claim: Claim) -> list[Question]:
        return [
            q for q in self.questions.values()
            if any(a.id == claim.id for a in self.answers_to(q.id))
        ]

    def _answer_note(self, question: Question) -> str:
        answers = self.answers_to(question.id)
        return f"answered by {', '.join(c.id for c in answers)}" if answers else "open"

    def _unresolved(self, path: list[Claim]) -> list[tuple[Claim, Question]]:
        """Questions about claims in a chain that the chain itself does not answer."""
        in_chain = {c.id for c in path}
        found: dict[str, tuple[Claim, Question]] = {}
        for claim in path:
            for q in self._questions_on(claim):
                answered_here = any(a.id in in_chain for a in self.answers_to(q.id))
                if not answered_here and q.id not in found:
                    found[q.id] = (claim, q)
        return list(found.values())

    def _record(self, actor: str, action: str, target: str, detail: str = "") -> None:
        self.log.append(
            {"at": self._clock(), "actor": actor, "action": action, "target": target, "detail": detail}
        )
