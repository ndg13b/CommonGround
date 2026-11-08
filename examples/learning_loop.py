"""Worked example: deciding what a model may learn from the graph, and re-checking after a change.

A person and a model build a graph in three levels: a principle (which only the
person may write), a concept, and facts with a conclusion that rests on them.
A plan sorts what is ready to be trained into the model's weights from what is
held back, and generates true/false probes to check the model before and after
training. Then a news report is corrected: the fact it supported is revised,
the conclusion resting on it is flagged for re-checking, and once both sides
have re-checked it, the new plan includes negative probes for what was revised
away and a ripple probe for the conclusion that changed.

Nothing is trained here. See docs/learning.md for the design this prototypes.

Run from the repo root:  python examples/learning_loop.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commonground import CommonGround, Level, Threshold, plan_consolidation  # noqa: E402


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def show_plan(graph: CommonGround, **options) -> None:
    plan = plan_consolidation(graph, **options)
    print("  Ready to train into weights: " + (", ".join(c.id for c in plan.ready) or "none"))
    for claim_id, reasons in plan.held.items():
        print(f"  Held back {claim_id}: {'; '.join(reasons)}")


def main() -> None:
    g = CommonGround()
    g.add_participant("model", "model")
    g.add_participant("nina", "human")
    g.add_source("doc1", "Acme annual report, suppliers section")
    g.add_source("doc2", "News report: VoltCo's Lyon plant recalls cells made in May")
    g.add_source("doc2-copy", "Syndicated copy of the same news report", origin="doc2")
    g.add_source("doc4", "Acme purchase order PO-1182: ship from VoltCo, Lyon")

    section("1. A principle, which only a person may write")
    principle = g.propose("nina", "Recall advice", "must put first", "customer safety",
                          "Company policy for anything we tell customers.", level="principle")
    print(f"  {principle.id}: {principle.triple()}  [principle, by nina]")
    try:
        g.propose("model", "Recall advice", "must minimise", "cost",
                  "Recalls are expensive.", level="principle")
    except ValueError as refusal:
        print(f"  The model tries to add a principle and is refused: {refusal}")
    g.respond(principle.id, "model", "agree", "Consistent with how recall notices should work.")

    section("2. A concept, then facts and a conclusion that rests on them")
    concept = g.propose("model", "A supplier recall", "affects",
                        "customers whose parts come from the recalled plant",
                        "This is what a recall means for a customer.", level=Level.CONCEPT)
    supply = g.propose("model", "Acme Corp", "buys batteries from", "VoltCo",
                       "Doc 1 names VoltCo as Acme's battery supplier.", ["doc1"])
    plant = g.propose("model", "VoltCo", "makes cells at", "Lyon plant",
                      "Acme's purchase order ships from Lyon.", ["doc4"],
                      qualifiers={"for customer": "Acme Corp"})
    recall = g.propose("model", "Lyon plant", "had", "May recall",
                       "The news report and its syndicated copy.", ["doc2", "doc2-copy"])
    conclusion = g.propose("model", "Acme Corp", "is affected by", "May recall",
                           "Follows from the supply chain, the recall and what a recall means.",
                           depends_on=[supply.id, plant.id, recall.id, concept.id])
    for claim in (concept, supply, plant, recall, conclusion):
        g.respond(claim.id, "nina", "agree")
        print(f"  {claim.id}: {claim.triple()}")
    print(f"  {conclusion.id} rests on {', '.join(conclusion.depends_on)}.")

    section("3. What may be trained into the model's weights?")
    show_plan(g)
    print("  With a stricter bar of two independent sources for facts:")
    show_plan(g, thresholds={Level.FACT: Threshold(sources=2)})
    print(f"  ({recall.id} cites doc2 and doc2-copy, but the copy repeats doc2, so they count as one.)")

    section("4. The news report is corrected: the recall was at Porto, not Lyon")
    g.add_source("doc5", "Corrected news report: the May recall was at VoltCo's Porto plant")
    corrected = g.revise(recall.id, "model", "The report was corrected.",
                         subject="Porto plant", rationale="The corrected news report.",
                         sources=["doc5"])
    print(f"  {corrected.id}: {corrected.triple()}  (revises {recall.id})")
    print("  Claims to re-check:")
    for claim, reasons in g.needs_recheck():
        print(f"  {claim.id}: {claim.triple()}: {'; '.join(reasons)}")
    show_plan(g)

    section("5. Both sides re-check the branch")
    g.respond(corrected.id, "nina", "agree")
    new_conclusion = g.revise(
        conclusion.id, "model", "Re-checked against the corrected report.",
        relation="is not affected by",
        rationale="Acme's cells come from Lyon, and the recall was at Porto.",
        depends_on=[plant.id, corrected.id, concept.id],
    )
    g.respond(new_conclusion.id, "nina", "agree")
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
