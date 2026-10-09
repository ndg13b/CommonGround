"""Worked example: a model and a human build and correct a graph together.

The model extracts claims from three documents. The human spots a gap the model
couldn't see (VoltCo has a second plant), the model finds evidence that closes
it, and the conclusion ends up resting only on links both sides accept.

No LLM is called here: the model's turns are scripted so the example is
reproducible. In a real system those turns would come from an LLM that reads
`graph.to_context()` and replies with propose / respond / revise actions.

Run from the repo root:  python examples/acme_recall.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commonground import CommonGround  # noqa: E402

QUESTION = ("Acme Corp", "May recall")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def ask(graph: CommonGround) -> None:
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

    section("1. The model extracts claims from three documents")
    c1 = g.propose("model", "Acme Corp", "buys batteries from", "VoltCo",
                   "Doc 1 names VoltCo as Acme's battery supplier.", ["doc1"])
    c2 = g.propose("model", "VoltCo", "makes cells at", "Lyon plant",
                   "Doc 2 describes VoltCo's cell production in Lyon.", ["doc2"])
    c3 = g.propose("model", "Lyon plant", "had", "May recall",
                   "Doc 3 reports 10,000 cells recalled from Lyon in May.", ["doc3"])
    for c in (c1, c2, c3):
        print(f"  {c.id}: {c.triple()}")

    section("2. The human reviews them")
    g.respond(c1.id, "nina", "agree")
    g.respond(c3.id, "nina", "agree")
    g.respond(c2.id, "nina", "unsure",
              "VoltCo also runs a plant in Porto. Which plant makes Acme's cells?")
    print("  nina agrees with c1 and c3, and is unsure about c2.")

    section("3. Question: is Acme Corp affected by the May recall?")
    ask(g)

    section("4. The human adds what she knows; the model accepts it")
    c4 = g.propose("nina", "VoltCo", "makes cells at", "Porto plant",
                   "Our supplier register lists a second VoltCo plant in Porto.",
                   ["supplier-register"])
    g.respond(c4.id, "model", "agree")
    print(f"  {c4.id}: {c4.triple()}  [{c4.status.value}]")

    section("5. The model finds evidence and revises its claim")
    c5 = g.revise(
        c2.id, "model",
        reason="Narrowing c2 so it answers nina's question about which plant.",
        relation="ships Acme's orders from",
        rationale="Acme's purchase order lists 'ship from: VoltCo, Lyon'.",
        sources=["doc4 (purchase order)"],
    )
    g.respond(c5.id, "nina", "agree")
    print(g.why(c5.id))

    section("6. Same question, asked again")
    ask(g)

    section("Still open")
    open_items = g.open_questions()
    print("  Nothing left to discuss." if not open_items else "\n".join(
        f"  {c.id}: {c.triple()} [{c.status.value}]" for c in open_items))

    section("What an LLM would see as context")
    print(g.to_context())

    section("Mermaid diagram (paste into mermaid.live or a GitHub README)")
    print(g.to_mermaid())

    section("Conversation log")
    for entry in g.log:
        print(f"  {entry['actor']:>5} {entry['action']:<8} {entry['claim']:<3} {entry['detail']}")

    out = Path(__file__).with_name("acme_recall.json")
    g.save(out)
    print(f"\nSaved the full graph and history to {out.name}")


if __name__ == "__main__":
    main()
