"""Worked example: a model and a human build and correct a graph together.

The model extracts claims from three documents. The human agrees with the ones
the question depends on, but asks something the documents don't answer:
VoltCo has a second plant, so which one makes Acme's cells? The model finds
evidence, refines its claim without discarding the original, and offers the
refinement as the answer. The conclusion ends up resting only on accepted
claims, and the chain that skips the answer says so.

No LLM is called here: the model's turns are scripted so the example is
reproducible. In a real system those turns would come from an LLM that reads
`graph.to_context()` and replies with propose / respond / ask / answer /
refine / revise actions.

Run from the repo root:  python examples/acme_recall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commonground import CommonGround  # noqa: E402

QUESTION = ("Acme Corp", "May recall")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_question(graph: CommonGround) -> None:
    start, end = QUESTION
    paths = graph.find_paths(start, end)
    if not paths:
        print(f"No chain of claims links {start} to {end}.")
        return
    for i, path in enumerate(paths, 1):
        print(f"Chain {i}:")
        print(graph.explain_path(path))


def main() -> None:
    g = CommonGround()
    g.add_participant("model", "model")
    g.add_participant("nina", "human")
    g.add_source("doc1", "Acme annual report, suppliers section")
    g.add_source("doc2", "VoltCo company profile")
    g.add_source("doc3", "News report on the May battery recall")

    section("1. The model extracts claims from three documents")
    c1 = g.propose("model", "Acme Corp", "buys batteries from", "VoltCo",
                   "Doc 1 names VoltCo as Acme's battery supplier.", ["doc1"])
    c2 = g.propose("model", "VoltCo", "makes cells at", "Lyon plant",
                   "Doc 2 describes VoltCo's cell production in Lyon.", ["doc2"])
    c3 = g.propose("model", "VoltCo", "was founded in", "1998",
                   "Doc 2 gives VoltCo's founding year.", ["doc2"])
    c4 = g.propose("model", "Lyon plant", "had", "May recall",
                   "Doc 3 reports 10,000 cells recalled from Lyon in May.", ["doc3"])
    for c in (c1, c2, c3, c4):
        print(f"  {c.id}: {c.triple()}")

    section("2. What should the human check first?")
    print("  Ranked by whether 'is Acme Corp affected by the May recall?' depends on it:")
    for c in g.review_queue(focus=[QUESTION]):
        print(f"  {c.id}: {c.triple()}")

    section("3. The human reviews the claims the question depends on")
    g.respond(c1.id, "nina", "agree")
    g.respond(c2.id, "nina", "agree")
    g.respond(c4.id, "nina", "agree")
    q1 = g.ask("nina", "VoltCo also runs a plant in Porto. Which plant makes Acme's cells?",
               about=[c2.id])
    print("  nina agrees with c1, c2 and c4. She doesn't doubt c2, but it doesn't say")
    print(f"  which plant Acme's cells come from, so she asks {q1.id} about it.")
    print("  She leaves c3 for later: the question doesn't depend on it.")

    section("4. Question: is Acme Corp affected by the May recall?")
    check_question(g)

    section("5. The human adds what she knows; the model checks it and agrees")
    g.add_source("supplier-register", "Acme's internal supplier register", by="nina")
    c5 = g.propose("nina", "VoltCo", "makes cells at", "Porto plant",
                   "Our supplier register lists a second VoltCo plant in Porto.",
                   ["supplier-register"])
    g.respond(c5.id, "model", "agree",
              "The register is Acme's own record of VoltCo's sites, and docs 1-3 don't contradict it.")
    print(f"  {c5.id}: {c5.triple()}  [{c5.status.value}]")

    section("6. The model finds evidence, refines its claim and answers the question")
    g.add_source("doc4", "Acme purchase order PO-1182")
    c6 = g.refine(
        c2.id, "model",
        reason=f"Narrows c2 to Acme's orders, to answer {q1.id}.",
        qualifiers={"for customer": "Acme Corp"},
        rationale="Acme's purchase order lists 'ship from: VoltCo, Lyon'.",
        sources=["doc4"],
    )
    g.answer(q1.id, "model", c6.id)
    g.respond(c6.id, "nina", "agree")
    print(g.why(c6.id))
    print()
    print(g.why(c2.id))

    section("7. Same question, asked again")
    check_question(g)

    section("Still open")
    questions = g.open_questions()
    print("  Questions: " + ("none open." if not questions else ""))
    for q in questions:
        print(f"  {q.id}: {q.text}")
    print("  Claims awaiting review:")
    for c in g.review_queue():
        print(f"  {c.id}: {c.triple()} [{c.status.value}]")

    section("What an LLM would see as context")
    print(g.to_context())

    section("Mermaid diagram (paste into mermaid.live or a GitHub README)")
    print(g.to_mermaid())

    section("Conversation log")
    for entry in g.log:
        print(f"  {entry['actor']:>6} {entry['action']:<10} {entry['target']:<17} {entry['detail']}")

    out = Path(__file__).with_name("acme_recall.json")
    g.save(out)
    print(f"\nSaved the full graph and history to {out.name}")


if __name__ == "__main__":
    main()
