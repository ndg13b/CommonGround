"""Worked example: deciding what a model may learn from the graph, and re-checking after a change.

A person and two different models build a graph in three levels: a principle
(which only the person may write), a concept, and facts with claims that rest
on them. A plan sorts what is ready to be trained into a model's weights from
what is held back, and generates true/false probes to check the model before
and after training. Then a news report is corrected: the fact it supported is
revised, and everything resting on it is flagged. One claim still holds and
is re-checked in place; the other no longer holds and is revised. The new
plan includes negative probes for what was revised away and ripple probes
for the claims that rest on the change.

Nothing is trained here. See docs/learning.md for the design this prototypes,
and experiments/ for a small training run that uses it.

Run from the repo root:  python examples/learning_loop.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commonground import Claim, CommonGround, Level, plan_consolidation  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def show_plan(graph: CommonGround, **options) -> None:
    plan = plan_consolidation(graph, **options)
    print("  Ready to train into weights: " + (", ".join(c.id for c in plan.ready) or "none"))
    for claim_id, reasons in plan.held.items():
        print(f"  Held back {claim_id}: {'; '.join(reasons)}")


def build() -> CommonGround:
    """The graph after steps 1 and 2: every claim agreed, the recall resting on one source."""
    g = CommonGround()
    g.add_participant("model", "model")
    g.add_participant("checker", "model", identity="a different model")
    g.add_participant("nina", "human")
    g.add_source("doc1", "Acme annual report, suppliers section", reliability="high")
    g.add_source("doc2", "News report: VoltCo's Lyon plant recalls cells made in May",
                 reliability="moderate")
    g.add_source("doc2-copy", "Syndicated copy of the same news report", origin="doc2")
    g.add_source("doc3", "VoltCo company profile: cell production in Lyon", reliability="moderate")
    g.add_source("doc4", "Acme purchase order PO-1182: VoltCo cells, ship from Lyon",
                 reliability="high")

    principle = g.state("nina", "Advice about recalls must put customer safety first.",
                        "Company policy for anything we tell customers.", level="principle")
    g.respond(principle.id, "model", "agree", "Consistent with how recall notices should work.")
    g.respond(principle.id, "checker", "agree", "A sound rule for recall advice.")
    concept = g.state("model", "A supplier's recall affects a customer whose parts come "
                      "from the recalled plant.",
                      "This is what a recall means for a customer.", level=Level.CONCEPT)
    supply = g.propose("model", "Acme Corp", "buys batteries from", "VoltCo",
                       "The annual report and a purchase order both say so.", ["doc1", "doc4"])
    plant = g.propose("model", "VoltCo", "makes cells at", "Lyon plant",
                      "Acme's purchase order ships from Lyon; VoltCo's profile agrees.",
                      ["doc4", "doc3"], qualifiers={"for customer": "Acme Corp"})
    recall = g.propose("model", "Lyon plant", "had", "May recall",
                       "The news report and its syndicated copy.", ["doc2", "doc2-copy"])
    notice = g.propose("model", "VoltCo", "must notify", "its customers",
                       "A recall at any of its plants obliges VoltCo to notify customers.",
                       depends_on=[recall.id])
    conclusion = g.propose("model", "Acme Corp", "is affected by", "May recall",
                           "Follows from the supply chain, the recall and what a recall means.",
                           depends_on=[supply.id, plant.id, recall.id, concept.id])
    for claim in (concept, supply, plant, recall, notice, conclusion):
        g.respond(claim.id, "nina", "agree")
    return g


def confirm_recall(g: CommonGround) -> None:
    """VoltCo's press release arrives, and nina cites it for the recall."""
    g.add_source("doc5", "VoltCo press release confirming a May recall", reliability="high")
    g.respond("c5", "nina", "agree", "VoltCo's press release confirms it.", sources=["doc5"])


def correct_report(g: CommonGround) -> Claim:
    """The news report is corrected, and the recall is revised to say Porto."""
    g.add_source("doc6", "Corrected news report: the May recall was at VoltCo's Porto plant",
                 reliability="moderate")
    corrected = g.revise("c5", "model", "The report was corrected.",
                         subject="Porto plant", rationale="The corrected report and the press release.",
                         sources=["doc6", "doc5"])
    g.respond(corrected.id, "nina", "agree")
    return corrected


def recheck_branch(g: CommonGround, corrected: Claim) -> Claim:
    """The notice still holds, so it is re-checked in place; the conclusion is revised."""
    g.reaffirm("c6", "nina", "VoltCo still had a recall, wherever it was.")
    new_conclusion = g.revise(
        "c7", "model", "Re-checked against the corrected report.",
        relation="is not affected by",
        rationale="Acme's cells come from Lyon, and the recall was at Porto.",
        depends_on=["c4", corrected.id, "c2"],
    )
    g.respond(new_conclusion.id, "nina", "agree")
    return new_conclusion


def main() -> None:
    g = build()
    claims = list(g.claims.values())
    principle, concept, supply, plant, recall, notice, conclusion = claims

    section("1. A principle, written as a sentence, which only a person may write")
    print(f"  {principle.id}: {principle.triple()}  [principle, by nina]")
    try:
        g.state("model", "Recall advice must keep costs down.", "Recalls are expensive.",
                level="principle")
    except ValueError as refusal:
        print(f"  The model tries to add a principle and is refused: {refusal}")
    print("  nina, the model and a second, different model all agree with it.")

    section("2. A concept, facts, and claims resting on them")
    for claim in claims[1:]:
        rests = f"  (rests on {', '.join(claim.depends_on)})" if claim.depends_on else ""
        print(f"  {claim.id}: {claim.triple()}{rests}")

    section("3. What may be trained into the model's weights?")
    show_plan(g)
    print(f"  ({recall.id} cites doc2 and doc2-copy, but the copy repeats doc2, so they count as one.)")
    confirm_recall(g)
    print("  VoltCo's press release arrives, and nina cites it:")
    show_plan(g)

    section("4. The news report is corrected: the recall was at Porto, not Lyon")
    corrected = correct_report(g)
    print(f"  {corrected.id}: {corrected.triple()}  (revises {recall.id})")
    print("  Claims to re-check:")
    for claim, reasons in g.needs_recheck():
        print(f"  {claim.id}: {claim.triple()}: {'; '.join(reasons)}")

    section("5. Both claims are re-checked")
    new_conclusion = recheck_branch(g, corrected)
    print(f"  {notice.id} still holds, so nina re-checks it in place: it keeps its agreement.")
    print(f"  {conclusion.id} no longer holds, so the model revises it:")
    print(g.why(new_conclusion.id))
    print("  Claims to re-check: " + (", ".join(c.id for c, _ in g.needs_recheck()) or "none"))

    section("6. The new plan, with probes to run before and after training")
    plan = plan_consolidation(g, changed=[corrected.id])
    show_plan(g, changed=[corrected.id])
    for probe in plan.probes:
        verdict = "true " if probe.expected else "false"
        print(f"  {probe.kind:<8} expect {verdict}  {probe.statement}")

    section("What a model would see as context")
    print(g.to_context())

    out = Path(__file__).with_name("learning_loop_plan.json")
    out.write_text(json.dumps({
        "ready": [{"id": c.id, "level": c.level.value, "claim": c.triple(),
                   "rationale": c.rationale, "sources": c.sources} for c in plan.ready],
        "held": plan.held,
        "probes": [asdict(p) for p in plan.probes],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved the plan (training candidates and probes) to {out.name}")


if __name__ == "__main__":
    main()
