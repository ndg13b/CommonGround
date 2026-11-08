"""Deciding what may be trained into a model's weights, and how to check that it worked.

The graph is the source of truth, and a model's weights are meant to be a
rebuildable copy of the parts of it that meet a bar. plan_consolidation()
applies that bar: it sorts claims into those ready to be trained in and those
held back, with the reasons. The bar is set per level, so principles and
concepts can be held to more than facts.

The plan also carries probes: true/false statements generated from the graph,
to put to a model before and after training. Core probes (principles and
concepts) must pass every time, so an update that damages the core can be
rejected. Negative probes check that what was revised away is no longer
believed, and ripple probes check that a change reached the claims resting on it.

Nothing here trains a model. It produces the inputs a training and evaluation
pipeline would need, and the record of why each claim was or wasn't included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

from .graph import (
    LEVELS,
    Claim,
    CommonGround,
    Level,
    Stance,
    Status,
    _content,
    _qualifier_text,
)


@dataclass(frozen=True)
class Threshold:
    """How much a claim needs before it may be trained into weights."""

    voices: int = 2   # distinct identities agreeing with it, counting its proposer
    humans: int = 0   # how many of those identities must be human
    sources: int = 0  # independent sources cited, unless the claim rests on other claims


# Facts need evidence from outside the conversation; concepts and principles
# need a person among the voices agreeing to them. Raise these as stakes rise.
DEFAULT_THRESHOLDS: Mapping[Level, Threshold] = MappingProxyType({
    Level.PRINCIPLE: Threshold(voices=2, humans=1),
    Level.CONCEPT: Threshold(voices=2, humans=1),
    Level.FACT: Threshold(voices=2, sources=1),
})


@dataclass
class Probe:
    """A statement to put to a model, and whether it should judge it true.

    Each ready claim gets one probe: "core" for principles and concepts, else
    "ripple" if it rests on a claim that just changed, else "fact". A claim
    revised away gets a "negative" probe once its replacement is ready.
    """

    kind: str  # "core", "ripple", "fact" or "negative"
    claim: str
    statement: str
    expected: bool


@dataclass
class ConsolidationPlan:
    ready: list[Claim] = field(default_factory=list)            # may be trained into weights
    held: dict[str, list[str]] = field(default_factory=dict)    # claim id -> why it may not
    probes: list[Probe] = field(default_factory=list)           # checks to run before and after


def plan_consolidation(
    graph: CommonGround,
    thresholds: Optional[Mapping[Level, Threshold]] = None,
    changed: Iterable[str] = (),
) -> ConsolidationPlan:
    """Sort active claims into ready and held, and generate probes for the ready ones.

    A claim is ready when it is accepted, meets its level's threshold, has no
    open question on it, doesn't need re-checking, and everything it rests on
    is ready too. `changed` names claims that were just changed: ready claims
    resting on them get ripple probes.
    """
    bar = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    recheck = {c.id: reasons for c, reasons in graph.needs_recheck()}
    verdicts: dict[str, list[str]] = {}

    def reasons_held(claim: Claim, visiting: frozenset[str]) -> list[str]:
        if claim.id in verdicts:
            return verdicts[claim.id]
        visiting = visiting | {claim.id}
        reasons = []
        if claim.status is not Status.ACCEPTED:
            reasons.append(f"not accepted ({claim.status.value})")
        reasons += _shortfall(graph, claim, bar[claim.level])
        reasons += [f"{q.id} is still open" for q in graph.questions_on(claim.id)
                    if not graph.answers_to(q.id)]
        reasons += recheck.get(claim.id, [])
        for d in claim.depends_on:
            dep = graph.current(d)
            # A dependency that is revised, unaccepted or itself needing re-checking
            # is already named above; this catches one held back for other reasons.
            settled = dep.id == d and dep.status is Status.ACCEPTED and d not in recheck
            if settled and d not in visiting and reasons_held(dep, visiting):
                reasons.append(f"rests on {d}, which isn't ready")
        verdicts[claim.id] = reasons
        return reasons

    plan = ConsolidationPlan()
    for level in LEVELS:
        for claim in graph.active_claims():
            if claim.level is level:
                reasons = reasons_held(claim, frozenset())
                if reasons:
                    plan.held[claim.id] = reasons
                else:
                    plan.ready.append(claim)
    plan.probes = _probes(graph, plan.ready, changed)
    return plan


def _shortfall(graph: CommonGround, claim: Claim, threshold: Threshold) -> list[str]:
    """How a claim falls short of a threshold; empty if it doesn't."""
    agreeing = [(name, p) for name, p in claim.positions.items() if p.stance is Stance.AGREE]
    voices = {p.identity for _, p in agreeing}
    humans = {p.identity for name, p in agreeing if graph.participants[name].kind == "human"}
    cited = list(claim.sources) + [s for _, p in agreeing for s in p.sources]
    origins = {graph.origin_of(s) for s in cited}
    reasons = []
    if len(voices) < threshold.voices:
        reasons.append(f"needs {threshold.voices} agreeing voices, has {len(voices)}")
    if len(humans) < threshold.humans:
        reasons.append(f"needs {threshold.humans} human voice(s), has {len(humans)}")
    if not claim.depends_on and len(origins) < threshold.sources:
        reasons.append(f"needs {threshold.sources} independent source(s), has {len(origins)}")
    return reasons


def _statement(claim: Claim) -> str:
    return f"{claim.subject} {claim.relation} {claim.object}{_qualifier_text(claim.qualifiers)}."


def _probes(graph: CommonGround, ready: list[Claim], changed: Iterable[str]) -> list[Probe]:
    ripple = {c.id for d in changed for c in graph.dependents(d)}
    ready_ids = {c.id for c in ready}
    probes = []
    for claim in ready:
        if claim.level is not Level.FACT:
            kind = "core"
        elif claim.id in ripple:
            kind = "ripple"
        else:
            kind = "fact"
        probes.append(Probe(kind, claim.id, _statement(claim), True))
    # What was revised away should no longer be believed, once its replacement is ready.
    for claim in graph.claims.values():
        if claim.status is Status.SUPERSEDED:
            current = graph.current(claim.id)
            if current.id in ready_ids and _content(current) != _content(claim):
                probes.append(Probe("negative", claim.id, _statement(claim), False))
    return probes
