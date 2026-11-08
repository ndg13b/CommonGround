"""Levels, rests-on links, re-checking, source ratings, and planning what may be trained in."""

import pytest

from commonground import CommonGround, Level, Reliability, Status, Threshold, plan_consolidation

CLOCK = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def g() -> CommonGround:
    graph = CommonGround(clock=lambda: CLOCK)
    graph.add_participant("model", "model")
    graph.add_participant("nina", "human")
    graph.add_participant("checker", "model", identity="a different model")
    graph.add_source("doc1", "supplier list", reliability="high")
    graph.add_source("doc2", "news report on the recall", reliability="moderate")
    graph.add_source("doc2-copy", "syndicated copy of the news report", origin="doc2")
    graph.add_source("doc3", "corrected news report", reliability="moderate")
    graph.add_source("doc4", "VoltCo press release", reliability="high")
    graph.add_source("doc5", "forum post", reliability="low")
    graph.add_source("doc6", "purchase order", reliability="high")
    graph.add_source("doc7", "blog post")
    return graph


def agree(g: CommonGround, *claims):
    """Have whoever didn't propose each claim agree with it."""
    for c in claims:
        other = "model" if g.participants[c.proposed_by].kind == "human" else "nina"
        g.respond(c.id, other, "agree", "checked")


def recall_case(g: CommonGround):
    """A principle, a concept, two facts and a conclusion resting on them, all agreed."""
    principle = g.state("nina", "Advice about recalls must put customer safety first.",
                        "Company policy.", level="principle")
    concept = g.propose("model", "A supplier recall", "affects",
                        "customers whose parts come from the recalled plant",
                        "How recalls work.", level=Level.CONCEPT)
    supply = g.propose("model", "Acme", "gets cells from", "Lyon plant", "docs 1 and 6", ["doc1", "doc6"])
    recall = g.propose("model", "Lyon plant", "had", "May recall", "docs 2 and 4", ["doc2", "doc4"])
    conclusion = g.propose("model", "Acme", "is affected by", "May recall",
                           "Follows from the supply, the recall and how recalls work.",
                           depends_on=[supply.id, recall.id, concept.id])
    agree(g, principle, concept, supply, recall, conclusion)
    g.respond(principle.id, "checker", "agree", "Safety first is right for recall advice.")
    return principle, concept, supply, recall, conclusion


# --------------------------------------------------------------------- levels

def test_claims_are_facts_unless_told_otherwise(g):
    assert g.propose("model", "A", "r", "B", "because").level is Level.FACT


def test_only_humans_author_principles(g):
    with pytest.raises(ValueError):
        g.propose("model", "A", "must", "B", "because", level="principle")
    principle = g.propose("nina", "A", "must", "B", "because", level="principle")
    fact = g.propose("model", "C", "r", "D", "because")
    for change in (g.revise, g.refine):
        with pytest.raises(ValueError):
            change(principle.id, "model", "softer", qualifiers={"unless": "costly"})
    with pytest.raises(ValueError):
        g.revise(fact.id, "model", "promote it", level="principle")
    narrower = g.refine(principle.id, "nina", "narrower", qualifiers={"for": "recalls"})
    assert narrower.level is Level.PRINCIPLE


# ------------------------------------------------------------------ sentences

def test_claims_can_be_written_as_sentences(g):
    principle = g.state("nina", "Advice about recalls must put customer safety first.",
                        "Company policy.", level="principle")
    assert principle.triple() == "Advice about recalls must put customer safety first"
    assert principle.subject == principle.relation == principle.object == ""
    assert "- c1: Advice about recalls must put customer safety first | proposed | inferred" in (
        g.to_context()
    )
    with pytest.raises(ValueError):
        g.state("nina", "  ", "why")
    with pytest.raises(ValueError):
        g.propose("model", "", "r", "B", "use state() for sentences")


def test_sentences_are_not_edges(g):
    g.state("nina", "Customers come first.", "Policy.")
    g.propose("model", "A", "r", "B", "because")
    assert g.relations_in_use() == ["r"]
    assert g.to_mermaid().count("-->") + g.to_mermaid().count(".->") == 1
    assert g.find_paths("", "") == []


def test_vocabulary_does_not_apply_to_sentences():
    g = CommonGround(vocabulary=["buys from"])
    g.add_participant("nina", "human")
    g.state("nina", "Customers come first.", "Policy.")


def test_sentences_are_changed_with_text(g):
    s1 = g.state("nina", "Customers come first.", "Policy.", level="principle")
    with pytest.raises(ValueError):
        g.revise(s1.id, "nina", "reword", subject="Customers")
    s2 = g.revise(s1.id, "nina", "reword", text="Customer safety comes first.")
    assert s2.text == "Customer safety comes first." and s2.level is Level.PRINCIPLE
    narrower = g.refine(s2.id, "nina", "narrow", qualifiers={"in": "recalls"})
    assert narrower.triple() == "Customer safety comes first (in: recalls)"
    with pytest.raises(ValueError):
        g.refine(s2.id, "nina", "no change")
    fact = g.propose("model", "A", "r", "B", "because")
    assert g.revise(fact.id, "nina", "say it plainly", text="A relates to B.").text == "A relates to B."


# --------------------------------------------------------------- dependencies

def test_dependencies_must_be_current_claims(g):
    a = g.propose("model", "A", "r", "B", "because")
    with pytest.raises(KeyError):
        g.propose("model", "B", "r", "C", "because", depends_on=["c99"])
    g.revise(a.id, "model", "fix", obj="B2")
    with pytest.raises(ValueError):
        g.propose("model", "B", "r", "C", "because", depends_on=[a.id])


def test_a_change_cannot_rest_on_the_claim_it_changes(g):
    a = g.propose("model", "A", "r", "B", "because")
    with pytest.raises(ValueError):
        g.revise(a.id, "model", "fix", depends_on=[a.id])
    with pytest.raises(ValueError):
        g.refine(a.id, "model", "narrow", qualifiers={"x": "1"}, depends_on=[a.id])


def test_revisions_inherit_the_current_dependencies(g):
    _, concept, supply, recall, conclusion = recall_case(g)
    new_recall = g.revise(recall.id, "model", "wrong month", obj="June recall")
    again = g.revise(conclusion.id, "model", "re-checked")
    assert again.depends_on == [supply.id, new_recall.id, concept.id]


def test_dependents_follow_chains_and_revisions(g):
    a = g.propose("model", "A", "r", "B", "because")
    b = g.propose("model", "B", "r", "C", "because", depends_on=[a.id])
    c = g.propose("model", "C", "r", "D", "because", depends_on=[b.id])
    assert [x.id for x in g.dependents(a.id)] == [b.id, c.id]
    a2 = g.revise(a.id, "model", "fix", obj="B2")
    assert [x.id for x in g.dependents(a2.id)] == [b.id, c.id]  # they rest on its earlier version


def test_circular_support_is_refused(g):
    a = g.propose("model", "A", "r", "B", "because")
    b = g.propose("model", "B", "r", "C", "because", depends_on=[a.id])
    c = g.propose("model", "C", "r", "D", "because", depends_on=[b.id])
    for start in (b, c):
        with pytest.raises(ValueError):
            g.revise(a.id, "model", "loop", depends_on=[start.id])
    assert g.revise(a.id, "model", "fine", obj="B2").depends_on == []


def test_loaded_cycles_terminate(g):
    a = g.propose("model", "A", "r", "B", "because")
    b = g.propose("model", "B", "r", "C", "because", depends_on=[a.id])
    a.depends_on = [b.id]  # as a hand-edited file might contain
    assert [x.id for x in g.dependents(a.id)] == [b.id]
    assert {c.id for c, _ in g.needs_recheck()} == {a.id, b.id}
    assert plan_consolidation(g).ready == []


# ----------------------------------------------------------------- re-checking

def test_a_change_flags_the_branch_resting_on_it(g):
    *_, recall, conclusion = recall_case(g)
    notice = g.propose("model", "Acme", "should notify", "its customers", "Follows from c5.",
                       depends_on=[conclusion.id])
    agree(g, notice)
    assert g.needs_recheck() == []
    new_recall = g.revise(recall.id, "model", "the recall was at Porto",
                          subject="Porto plant", sources=["doc3"])
    flagged = {c.id: reasons for c, reasons in g.needs_recheck()}
    assert flagged == {
        conclusion.id: [f"rests on {recall.id}, which was revised as {new_recall.id}"],
        notice.id: [f"rests on {conclusion.id}, which needs re-checking"],
    }


def test_disputing_a_claim_flags_what_rests_on_it(g):
    *_, supply, _, conclusion = recall_case(g)
    g.respond(supply.id, "nina", "disagree", "Acme switched to Porto")
    assert g.needs_recheck() == [(conclusion, [f"rests on {supply.id}, which is disputed"])]


def test_reaffirming_clears_the_flag(g):
    *_, recall, conclusion = recall_case(g)
    new_recall = g.revise(recall.id, "model", "better rationale", rationale="doc 2 and its copy")
    g.respond(new_recall.id, "nina", "agree")
    assert [c.id for c, _ in g.needs_recheck()] == [conclusion.id]
    again = g.revise(conclusion.id, "model", "re-checked against the new version")
    g.respond(again.id, "nina", "agree")
    assert g.needs_recheck() == []


def test_reaffirming_keeps_the_claim_and_its_agreement(g):
    *_, recall, conclusion = recall_case(g)
    notice = g.propose("model", "VoltCo", "must notify", "its customers", "A recall happened.",
                       depends_on=[recall.id])
    agree(g, notice)
    new_recall = g.revise(recall.id, "model", "the recall was at Porto",
                          subject="Porto plant", sources=["doc3", "doc4"])
    g.respond(new_recall.id, "nina", "agree")
    assert {c.id for c, _ in g.needs_recheck()} == {conclusion.id, notice.id}
    g.reaffirm(notice.id, "nina", "VoltCo still had a recall, wherever it was.")
    assert notice.depends_on == [new_recall.id]
    assert notice.status is Status.ACCEPTED and notice.id in g.claims
    assert [c.id for c, _ in g.needs_recheck()] == [conclusion.id]
    assert (f"Re-checked by nina ({recall.id} -> {new_recall.id}): "
            "VoltCo still had a recall, wherever it was.") in g.why(notice.id)
    assert g.log[-1]["action"] == "reaffirm"
    assert notice in plan_consolidation(g, changed=[new_recall.id]).ready


def test_who_may_reaffirm(g):
    a = g.propose("nina", "A", "r", "B", "because", ["doc1", "doc6"])
    mine = g.propose("model", "C", "r", "D", "because", depends_on=[a.id])
    theirs = g.propose("nina", "E", "r", "F", "because", depends_on=[a.id])
    concept = g.propose("nina", "K", "r", "L", "because", level="concept", depends_on=[a.id])
    agree(g, a, mine, theirs, concept)
    a2 = g.revise(a.id, "nina", "fix", obj="B2")
    g.respond(a2.id, "model", "agree", "checked the fix")
    with pytest.raises(ValueError):
        g.reaffirm(mine.id, "model", "still fine")  # its own claim
    g.add_participant("model-2", "model", identity="model")
    with pytest.raises(ValueError):
        g.reaffirm(mine.id, "model-2", "still fine")  # another instance of the same model
    with pytest.raises(ValueError):
        g.reaffirm(concept.id, "checker", "still fine")  # models don't re-check concepts
    with pytest.raises(ValueError):
        g.reaffirm(theirs.id, "checker", "  ")  # a reason is needed
    g.reaffirm(theirs.id, "checker", "Still follows from the fixed claim.")
    g.reaffirm(mine.id, "checker", "Still follows.")
    g.reaffirm(concept.id, "nina", "Still holds.")
    assert g.needs_recheck() == []


def test_reaffirming_needs_something_settled_to_rest_on(g):
    a = g.propose("model", "A", "r", "B", "because")
    b = g.propose("model", "B", "r", "C", "because", depends_on=[a.id])
    agree(g, a, b)
    with pytest.raises(ValueError):
        g.reaffirm(b.id, "nina", "nothing changed")  # nothing to re-check
    g.revise(a.id, "model", "fix", obj="B2")
    with pytest.raises(ValueError):
        g.reaffirm(b.id, "nina", "the fix isn't agreed yet")


def test_recheck_lists_most_entrenched_first(g):
    a = g.propose("model", "A", "r", "B", "because")  # never agreed
    fact = g.propose("model", "F", "r", "G", "because", depends_on=[a.id])
    concept = g.propose("model", "K", "r", "L", "because", level="concept", depends_on=[a.id])
    assert [c.id for c, _ in g.needs_recheck()] == [concept.id, fact.id]


def test_why_and_context_show_levels_and_support(g):
    _, concept, supply, recall, conclusion = recall_case(g)
    new_recall = g.revise(recall.id, "model", "wrong month", obj="June recall")
    why = g.why(conclusion.id)
    assert (f"Rests on {supply.id}, {recall.id} (since revised as {new_recall.id}), "
            f"{concept.id}.") in why
    assert f"Needs re-checking: rests on {recall.id}, which was revised as {new_recall.id}." in why
    assert f"Supports {conclusion.id}." in g.why(supply.id)
    assert g.why(concept.id).split("\n")[0].endswith("[accepted, concept]")
    context = g.to_context()
    assert (context.index("Principles (these take priority over everything below):")
            < context.index("Concepts:") < context.index("Facts:"))
    assert f"rests on {supply.id}, {recall.id} (since revised as {new_recall.id})" in context
    assert "needs re-checking" in context
    assert ("- doc2-copy: syndicated copy of the news report "
            "(supplied by the system; repeats doc2)") in context


def test_context_has_no_level_headings_when_all_claims_are_facts(g):
    g.propose("model", "A", "r", "B", "because")
    assert "Facts:" not in g.to_context()


def test_chains_report_claims_needing_recheck_and_rank_them_lower(g):
    *_, recall, conclusion = recall_case(g)
    g.revise(recall.id, "model", "wrong month", obj="June recall")
    other = g.propose("model", "Acme", "is exposed to", "May recall", "doc 3", ["doc3"])
    agree(g, other)
    paths = g.find_paths("Acme", "May recall")
    assert [[c.id for c in p] for p in paths] == [[other.id], [conclusion.id]]
    text = g.explain_path(paths[1])
    assert f"needs re-checking: rests on {recall.id}" in text
    assert f"{conclusion.id} needs re-checking, so this chain may rest on outdated support." in text
    assert "Every link is accepted" not in text


# --------------------------------------------------------------------- sources

def test_source_origins(g):
    assert g.origin_of("doc2-copy") == "doc2"
    g.add_source("doc2-copy-2", "a copy of the copy", origin="doc2-copy")
    assert g.origin_of("doc2-copy-2") == "doc2"
    assert g.origin_of("doc1") == "doc1"
    with pytest.raises(KeyError):
        g.add_source("x", "a copy of nothing", origin="missing")


# --------------------------------------------------------------- consolidation

def test_agreed_claims_with_enough_support_are_ready(g):
    case = recall_case(g)
    plan = plan_consolidation(g)
    assert plan.ready == list(case)  # principles first, then concepts, then facts
    assert plan.held == {}


def test_facts_need_two_reliable_sources_unless_they_rest_on_claims(g):
    *_, conclusion = recall_case(g)  # rests on claims, cites nothing, and is ready
    bare = g.propose("model", "A", "r", "B", "I remember it.")
    agree(g, bare)
    assert plan_consolidation(g).held == {
        bare.id: ["needs 2 independent source(s) rated moderate or better, has 0"]
    }
    assert conclusion in plan_consolidation(g).ready


def test_copies_count_as_one_source(g):
    fire = g.propose("model", "Lyon plant", "had", "a fire", "The news.", ["doc2", "doc2-copy"])
    agree(g, fire)
    assert plan_consolidation(g).held[fire.id] == [
        "needs 2 independent source(s) rated moderate or better, has 1"
    ]
    g.respond(fire.id, "nina", "agree", "VoltCo confirmed it", sources=["doc4"])
    assert fire.id not in plan_consolidation(g).held


def test_unrated_and_unreliable_sources_do_not_count(g):
    fact = g.propose("model", "A", "r", "B", "Three sources.", ["doc1", "doc5", "doc7"])
    agree(g, fact)
    assert plan_consolidation(g).held[fact.id] == [
        "needs 2 independent source(s) rated moderate or better, has 1"
    ]
    assert plan_consolidation(g, {Level.FACT: Threshold(sources=2)}).held == {}  # unrated counts here
    g.rate_source("doc7", "nina", "moderate", "A careful trade blog.")
    assert plan_consolidation(g).held == {}


def test_only_people_or_the_system_rate_sources(g):
    with pytest.raises(ValueError):
        g.add_source("x", "my own notes", by="model", reliability="high")
    with pytest.raises(ValueError):
        g.rate_source("doc7", "model", "high", "trust me")
    with pytest.raises(ValueError):
        g.rate_source("doc7", "nina", "high", "  ")
    assert g.add_source("y", "model notes", by="model").reliability is Reliability.UNRATED
    g.rate_source("doc7", "nina", Reliability.LOW, "Often wrong.")
    assert g.log[-1]["action"] == "rate source"
    assert "- doc7: blog post (supplied by the system; rated low)" in g.to_context()


def test_copies_count_with_their_originals_rating(g):
    g.add_source("doc5-copy", "a copy of the forum post", origin="doc5")
    g.rate_source("doc5-copy", "nina", "high", "Rated the copy by mistake.")
    fact = g.propose("model", "A", "r", "B", "Two sources.", ["doc1", "doc5-copy"])
    agree(g, fact)
    assert fact.id in plan_consolidation(g).held  # doc5-copy repeats doc5, which is rated low


def test_principles_need_three_voices_but_always_get_core_probes(g):
    principle = g.state("nina", "Be honest about uncertainty.", "Policy.", level="principle")
    g.respond(principle.id, "model", "agree", "Agreed.")
    plan = plan_consolidation(g)
    assert plan.held == {principle.id: ["needs 3 agreeing voices, has 2"]}
    assert [(p.kind, p.claim) for p in plan.probes] == [("core", principle.id)]
    g.respond(principle.id, "checker", "agree", "Agreed, as a different model.")
    assert plan_consolidation(g).ready == [principle]


def test_concepts_need_a_human_voice(g):
    g.add_participant("other-model", "model", identity="a different model")
    concept = g.propose("model", "K", "r", "L", "because", level="concept")
    g.respond(concept.id, "other-model", "agree", "sounds right")
    assert concept.status is Status.ACCEPTED
    assert plan_consolidation(g).held == {concept.id: ["needs 1 human voice(s), has 0"]}
    g.respond(concept.id, "nina", "agree")
    assert plan_consolidation(g).held == {}


def test_thresholds_can_ask_for_more(g):
    principle, *_ = recall_case(g)
    plan = plan_consolidation(g, {Level.PRINCIPLE: Threshold(voices=4, humans=2)})
    assert plan.held == {
        principle.id: ["needs 4 agreeing voices, has 3", "needs 2 human voice(s), has 1"]
    }


def test_unsettled_claims_are_held_with_reasons(g):
    _, concept, _, recall, conclusion = recall_case(g)
    q = g.ask("nina", "Which batches?", about=[recall.id])
    g.respond(concept.id, "nina", "unsure", "only if the batch was affected")
    held = plan_consolidation(g).held
    assert held[recall.id] == [f"{q.id} is still open"]
    assert held[concept.id] == [
        "not accepted (uncertain)",
        "needs 2 agreeing voices, has 1",
        "needs 1 human voice(s), has 0",
    ]
    assert held[conclusion.id] == [
        f"rests on {concept.id}, which is uncertain",
        f"rests on {recall.id}, which isn't ready",
    ]


def test_claims_needing_recheck_are_held(g):
    *_, recall, conclusion = recall_case(g)
    new_recall = g.revise(recall.id, "model", "wrong month", obj="June recall")
    held = plan_consolidation(g).held
    assert held[conclusion.id] == [f"rests on {recall.id}, which was revised as {new_recall.id}"]
    assert held[new_recall.id] == ["not accepted (proposed)", "needs 2 agreeing voices, has 1"]


# ---------------------------------------------------------------------- probes

def test_probes_cover_the_core_facts_ripples_and_what_was_revised_away(g):
    principle, concept, supply, recall, conclusion = recall_case(g)
    new_recall = g.revise(recall.id, "model", "wrong month", obj="June recall")
    g.respond(new_recall.id, "nina", "agree")
    again = g.revise(conclusion.id, "model", "re-checked", obj="June recall")
    g.respond(again.id, "nina", "agree")
    probes = plan_consolidation(g, changed=[new_recall.id]).probes
    assert {(p.kind, p.claim): (p.statement, p.expected) for p in probes} == {
        ("core", principle.id): ("Advice about recalls must put customer safety first.", True),
        ("core", concept.id): (
            "A supplier recall affects customers whose parts come from the recalled plant.", True
        ),
        ("fact", supply.id): ("Acme gets cells from Lyon plant.", True),
        ("fact", new_recall.id): ("Lyon plant had June recall.", True),
        ("ripple", again.id): ("Acme is affected by June recall.", True),
        ("negative", recall.id): ("Lyon plant had May recall.", False),
        ("negative", conclusion.id): ("Acme is affected by May recall.", False),
    }


def test_no_negative_probe_when_only_the_reasoning_changed(g):
    *_, recall, _ = recall_case(g)
    new_recall = g.revise(recall.id, "model", "better rationale", rationale="doc 2, page 3")
    g.respond(new_recall.id, "nina", "agree")
    assert all(p.kind != "negative" for p in plan_consolidation(g).probes)


def test_only_ready_claims_get_probes(g):
    bare = g.propose("model", "A", "r", "B", "I remember it.")
    agree(g, bare)
    assert plan_consolidation(g).probes == []


# --------------------------------------------------------------- serialization

def test_levels_dependencies_and_origins_survive_save_and_load(g, tmp_path):
    principle, concept, supply, recall, conclusion = recall_case(g)
    new_recall = g.revise(recall.id, "model", "wrong month", obj="June recall")
    g.respond(new_recall.id, "nina", "agree")
    g.reaffirm(conclusion.id, "nina", "June or May, Acme is affected.")
    g.rate_source("doc7", "nina", "moderate", "checked it")
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = CommonGround.load(path)
    assert loaded.to_dict() == g.to_dict()
    assert loaded.claims[concept.id].level is Level.CONCEPT
    assert loaded.claims[principle.id].text == principle.text
    assert loaded.claims[conclusion.id].depends_on == [supply.id, new_recall.id, concept.id]
    assert loaded.claims[conclusion.id].reaffirmed[0].replaced == {recall.id: new_recall.id}
    assert loaded.origin_of("doc2-copy") == "doc2"
    assert loaded.sources["doc7"].reliability is Reliability.MODERATE


def test_graphs_saved_by_0_2_still_load(g):
    a = g.propose("model", "A", "r", "B", "because", ["doc1"])
    data = g.to_dict()
    for c in data["claims"]:
        del c["level"], c["depends_on"], c["text"], c["reaffirmed"]
    for s in data["sources"]:
        del s["origin"], s["reliability"]
    loaded = CommonGround.from_dict(data)
    assert loaded.claims[a.id].level is Level.FACT
    assert loaded.claims[a.id].depends_on == []
