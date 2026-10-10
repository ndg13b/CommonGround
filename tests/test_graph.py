import pytest

from commonground import CommonGround, Stance, Status

CLOCK = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def g() -> CommonGround:
    graph = CommonGround(clock=lambda: CLOCK)
    graph.add_participant("model", "model")
    graph.add_participant("nina", "human")
    for doc in ("doc1", "doc2", "doc3", "doc4"):
        graph.add_source(doc, f"test document {doc}")
    return graph


def chain(g: CommonGround):
    c1 = g.propose("model", "Acme Corp", "buys from", "VoltCo", "doc 1 says so", ["doc1"])
    c2 = g.propose("model", "VoltCo", "makes cells at", "Lyon plant", "doc 2 says so", ["doc2"])
    c3 = g.propose("model", "Lyon plant", "had", "May recall", "doc 3 says so", ["doc3"])
    return c1, c2, c3


def chain_through(g: CommonGround, claim_id: str, start="Acme Corp", end="May recall"):
    return next(p for p in g.find_paths(start, end) if claim_id in [c.id for c in p])


def agreed_chain_with_question(g: CommonGround):
    """The worked example in miniature: every link accepted, a question asked about c2."""
    c1, c2, c3 = chain(g)
    for c in (c1, c2, c3):
        g.respond(c.id, "nina", "agree")
    q = g.ask("nina", "which plant makes Acme's cells?", about=[c2.id])
    return c1, c2, c3, q


# ---------------------------------------------------------------- claim rules

def test_new_claim_is_proposed_until_someone_else_agrees(g):
    c1, *_ = chain(g)
    assert c1.status is Status.PROPOSED
    g.respond(c1.id, "nina", "agree")
    assert c1.status is Status.ACCEPTED


def test_proposer_agreeing_again_does_not_accept_own_claim(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "model", "agree", "still think so")
    assert c1.status is Status.PROPOSED


def test_disagreement_outranks_agreement(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "nina", "disagree", "Acme switched suppliers last year")
    assert c1.status is Status.DISPUTED


def test_unsure_marks_claim_uncertain(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "nina", "unsure", "not sure this is current")
    assert c1.status is Status.UNCERTAIN


def test_changing_your_mind_replaces_your_position(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "nina", "disagree", "wrong supplier")
    g.respond(c1.id, "nina", "agree")
    assert c1.status is Status.ACCEPTED
    assert c1.positions["nina"].stance is Stance.AGREE


@pytest.mark.parametrize("stance", ["disagree", "unsure"])
def test_objections_require_a_reason(g, stance):
    c1, *_ = chain(g)
    with pytest.raises(ValueError):
        g.respond(c1.id, "nina", stance)


def test_claims_require_a_rationale(g):
    with pytest.raises(ValueError):
        g.propose("model", "A", "rel", "B", "   ")


def test_unknown_participant_rejected(g):
    with pytest.raises(KeyError):
        g.propose("stranger", "A", "rel", "B", "because")


def test_invalid_stance_rejected(g):
    c1, *_ = chain(g)
    with pytest.raises(ValueError):
        g.respond(c1.id, "nina", "maybe", "hmm")


def test_registering_a_name_again(g):
    assert g.add_participant("nina", "human") is g.participants["nina"]
    with pytest.raises(ValueError):
        g.add_participant("nina", "model")
    with pytest.raises(ValueError):
        g.add_participant("model", "model", identity="another model")


def test_system_is_not_a_participant_name(g):
    with pytest.raises(ValueError):
        g.add_participant("system", "human")


# ------------------------------------------------------------------ agreement

def test_model_agreement_needs_a_reason(g):
    c = g.propose("nina", "A", "rel", "B", "because")
    with pytest.raises(ValueError):
        g.respond(c.id, "model", "agree")
    g.respond(c.id, "model", "agree", "matches doc 1", sources=["doc1"])
    assert c.status is Status.ACCEPTED
    assert c.positions["model"].sources == ["doc1"]


def test_agreement_reason_rule_is_configurable():
    strict = CommonGround(agreement_reason_required=("human", "model"))
    strict.add_participant("model", "model")
    strict.add_participant("nina", "human")
    c = strict.propose("model", "A", "rel", "B", "because")
    with pytest.raises(ValueError):
        strict.respond(c.id, "nina", "agree")

    lax = CommonGround(agreement_reason_required=())
    lax.add_participant("model", "model")
    lax.add_participant("nina", "human")
    c = lax.propose("nina", "A", "rel", "B", "because")
    lax.respond(c.id, "model", "agree")
    assert c.status is Status.ACCEPTED


def test_agreement_reason_rule_only_takes_known_kinds():
    with pytest.raises(ValueError):
        CommonGround(agreement_reason_required=("robot",))


def test_participants_sharing_an_identity_count_as_one_voice(g):
    g.add_participant("model-2", "model", identity="model")
    g.add_participant("other-model", "model", identity="a different model")
    c1, c2, _ = chain(g)
    g.respond(c1.id, "model-2", "agree", "I read doc 1 too")
    assert c1.status is Status.PROPOSED
    g.respond(c1.id, "nina", "agree")
    assert c1.status is Status.ACCEPTED
    g.respond(c2.id, "other-model", "agree", "doc 2 says so")
    assert c2.status is Status.ACCEPTED


# -------------------------------------------------------------------- sources

def test_only_registered_sources_can_be_cited(g):
    with pytest.raises(KeyError):
        g.propose("model", "A", "rel", "B", "because", ["made-up-doc"])
    c1, *_ = chain(g)
    with pytest.raises(KeyError):
        g.respond(c1.id, "nina", "agree", "see this", sources=["made-up-doc"])


def test_sources_record_who_supplied_them(g):
    register = g.add_source("register", "supplier register", by="nina")
    assert register.added_by == "nina"
    assert g.sources["doc1"].added_by is None
    assert (g.log[0]["actor"], g.log[-1]["actor"]) == ("system", "nina")
    context = g.to_context()
    assert "doc1: test document doc1 (supplied by the system)" in context
    assert "register: supplier register (added by nina)" in context


def test_source_ids_are_unique(g):
    with pytest.raises(ValueError):
        g.add_source("doc1", "another document")


# ------------------------------------------------------------------ questions

def test_question_leaves_claim_status_alone(g):
    _, c2, _, q = agreed_chain_with_question(g)
    assert c2.status is Status.ACCEPTED
    assert g.open_questions() == [q]


def test_open_question_is_reported_on_the_chain(g):
    agreed_chain_with_question(g)
    text = g.explain_path(g.find_paths("Acme Corp", "May recall")[0])
    assert "q1 from nina (open): which plant makes Acme's cells?" in text
    assert "q1 on c2 is still open" in text
    assert "Every link is accepted" not in text


def test_question_is_answered_once_the_answer_is_accepted(g):
    _, c2, _, q = agreed_chain_with_question(g)
    c4 = g.refine(c2.id, "model", "narrowing", qualifiers={"for customer": "Acme Corp"},
                  rationale="purchase order", sources=["doc4"])
    g.answer(q.id, "model", c4.id)
    assert g.open_questions() == [q]
    g.respond(c4.id, "nina", "agree")
    assert g.open_questions() == []
    assert [c.id for c in g.answers_to(q.id)] == [c4.id]


def test_chain_that_skips_the_answer_says_so(g):
    _, c2, _, q = agreed_chain_with_question(g)
    c4 = g.refine(c2.id, "model", "narrowing", qualifiers={"for customer": "Acme Corp"},
                  rationale="purchase order", sources=["doc4"])
    g.answer(q.id, "model", c4.id)
    g.respond(c4.id, "nina", "agree")
    paths = g.find_paths("Acme Corp", "May recall")
    # equal length, so the chain that answers the question comes first
    assert [[c.id for c in p] for p in paths] == [["c1", "c4", "c3"], ["c1", "c2", "c3"]]
    best, other = (g.explain_path(p) for p in paths)
    assert "answers q1" in best
    assert "Every link is accepted and no question about them is open." in best
    assert "q1 on c2 was answered by c4, which is not in this chain" in other


def test_question_cannot_be_answered_by_the_claim_it_is_about(g):
    _, c2, _, q = agreed_chain_with_question(g)
    with pytest.raises(ValueError):
        g.answer(q.id, "model", c2.id)
    c4 = g.revise(c2.id, "model", "renamed", obj="Lyon Plant")
    with pytest.raises(ValueError):
        g.answer(q.id, "model", c4.id)  # the same claim, revised


def test_only_agreement_after_the_offer_answers(g):
    c1, _, _, q = agreed_chain_with_question(g)  # c1 was accepted before the question
    g.answer(q.id, "model", c1.id)
    assert g.open_questions() == [q]
    g.respond(c1.id, "model", "agree", "still true")  # the offerer can't settle it
    assert g.open_questions() == [q]
    g.respond(c1.id, "nina", "agree")
    assert g.open_questions() == []


def test_refinements_carry_questions_until_answered_in_the_chain(g):
    _, c2, _, q = agreed_chain_with_question(g)
    r = g.refine(c2.id, "model", "add a scope", qualifiers={"product": "cells"})
    g.respond(r.id, "nina", "agree")
    assert f"q1 on {r.id} is still open" in g.explain_path(chain_through(g, r.id))
    g.answer(q.id, "model", r.id)
    g.respond(r.id, "nina", "agree", "the purchase order confirms it")
    best = g.find_paths("Acme Corp", "May recall")[0]
    assert r in best
    text = g.explain_path(best)
    assert "answers q1 from nina: which plant makes Acme's cells?" in text
    assert "Every link is accepted and no question about them is open." in text


def test_questions_reach_refinements_of_earlier_versions(g):
    _, c2, _ = chain(g)
    for c in g.active_claims():
        g.respond(c.id, "nina", "agree")
    r = g.refine(c2.id, "model", "narrow", qualifiers={"for customer": "Acme"})
    g.respond(r.id, "nina", "agree")
    c5 = g.revise(c2.id, "model", "better rationale", rationale="doc 2, page 3")
    g.respond(c5.id, "nina", "agree")
    g.ask("nina", "which plant makes Acme's cells?", about=[c5.id])
    assert f"q1 on {r.id} is still open" in g.explain_path(chain_through(g, r.id))
    assert f"| refines c2 (since revised as {c5.id})\n" in g.to_context()
    assert f"Refines c2 (since revised as {c5.id})" in g.why(r.id)


def test_a_copy_of_the_questioned_claim_cannot_answer_it(g):
    _, c2, _, q = agreed_chain_with_question(g)
    r = g.refine(c2.id, "model", "narrow", qualifiers={"for customer": "Acme"})
    duplicate = g.revise(r.id, "model", "drop the scope", qualifiers={})
    assert duplicate.refines is None  # no longer narrower than c2
    with pytest.raises(ValueError):
        g.answer(q.id, "model", duplicate.id)


def test_answers_follow_revisions(g):
    _, c2, _, q = agreed_chain_with_question(g)
    c4 = g.refine(c2.id, "model", "narrowing", qualifiers={"for customer": "Acme Corp"})
    g.answer(q.id, "model", c4.id)
    c5 = g.revise(c4.id, "model", "wrong customer name", qualifiers={"for customer": "Acme Corp."})
    g.respond(c5.id, "nina", "agree")
    assert [c.id for c in g.answers_to(q.id)] == [c5.id]


def test_questions_carry_over_to_revisions(g):
    _, c2, _, _ = agreed_chain_with_question(g)
    c4 = g.revise(c2.id, "model", "wrong plant name", obj="Lyon Plant")
    assert "q1 from nina (open)" in g.why(c4.id)


def test_question_rules(g):
    c1, *_ = chain(g)
    with pytest.raises(ValueError):
        g.ask("nina", "  ", about=[c1.id])
    with pytest.raises(KeyError):
        g.ask("nina", "why?", about=["c99"])
    g.revise(c1.id, "model", "wrong supplier", obj="VoltCo SA")
    with pytest.raises(ValueError):
        g.ask("nina", "why?", about=[c1.id])
    q = g.ask("nina", "anything else?")
    with pytest.raises(ValueError):
        g.answer(q.id, "model", c1.id)
    with pytest.raises(KeyError):
        g.answer("q99", "model", "c2")


# ------------------------------------------------------------------ revisions

def test_revision_keeps_history_and_links_both_ways(g):
    _, c2, _ = chain(g)
    new = g.revise(c2.id, "model", "narrowing", relation="ships Acme orders from",
                   rationale="purchase order", sources=["doc4"])
    assert c2.status is Status.SUPERSEDED
    assert c2.superseded_by == new.id and new.supersedes == c2.id
    assert new.subject == "VoltCo" and new.object == "Lyon plant"
    assert new.sources == ["doc4"]
    assert c2.id in g.claims  # never deleted
    assert "Revises c2" in g.why(new.id)


def test_revising_a_refinement_keeps_the_link(g):
    _, c2, _ = chain(g)
    r = g.refine(c2.id, "model", "narrowing", qualifiers={"for customer": "Acme"})
    r2 = g.revise(r.id, "model", "full name", qualifiers={"for customer": "Acme Corp"})
    assert r2.refines == c2.id
    assert g.why(c2.id).endswith(f"Refined by {r2.id}.")
    assert "Refines c2" in g.why(r2.id)


def test_cannot_respond_to_or_change_superseded_claim(g):
    _, c2, _ = chain(g)
    g.revise(c2.id, "model", "narrowing", relation="ships from")
    with pytest.raises(ValueError):
        g.respond(c2.id, "nina", "agree")
    with pytest.raises(ValueError):
        g.revise(c2.id, "model", "again", relation="other")
    with pytest.raises(ValueError):
        g.refine(c2.id, "model", "again", qualifiers={"for customer": "Acme"})


def test_revision_requires_reason(g):
    _, c2, _ = chain(g)
    with pytest.raises(ValueError):
        g.revise(c2.id, "model", "", relation="x")


# ----------------------------------------------------------------- refinement

def test_refinement_keeps_the_original_active(g):
    _, c2, _ = chain(g)
    new = g.refine(c2.id, "model", "narrowing", qualifiers={"for customer": "Acme Corp"},
                   rationale="purchase order", sources=["doc4"])
    assert c2.status is Status.PROPOSED and new.refines == c2.id
    assert new.triple() == "VoltCo —makes cells at→ Lyon plant (for customer: Acme Corp)"
    assert new.sources == ["doc4"]
    assert "Refines c2" in g.why(new.id)
    assert f"Refined by {new.id}" in g.why(c2.id)


def test_refinement_adds_to_existing_qualifiers(g):
    c = g.propose("model", "A", "rel", "B", "because", qualifiers={"from": "2019"})
    r = g.refine(c.id, "model", "narrower", qualifiers={"until": "2023"})
    assert r.qualifiers == {"from": "2019", "until": "2023"}
    assert c.qualifiers == {"from": "2019"}


def test_refinement_must_narrow_the_claim(g):
    c = g.propose("model", "A", "rel", "B", "because", qualifiers={"from": "2019"})
    with pytest.raises(ValueError):
        g.refine(c.id, "model", "no change")
    with pytest.raises(ValueError):
        g.refine(c.id, "model", "no change", qualifiers={"from": "2019"})


def test_refinement_requires_reason(g):
    c1, *_ = chain(g)
    with pytest.raises(ValueError):
        g.refine(c1.id, "model", " ", qualifiers={"from": "2019"})


# ------------------------------------------------------------------ relations

def test_vocabulary_restricts_relations():
    g = CommonGround(vocabulary=["buys from"])
    g.add_participant("model", "model")
    g.propose("model", "A", "buys from", "B", "because")
    with pytest.raises(ValueError):
        g.propose("model", "A", "purchases from", "B", "because")
    assert "Allowed relations: buys from" in g.to_context()


def test_relations_in_use_listed_most_used_first(g):
    chain(g)
    g.propose("model", "VoltCo", "makes cells at", "Porto plant", "because")
    assert g.relations_in_use() == ["makes cells at", "buys from", "had"]
    assert "Relations in use (reuse one rather than inventing a synonym): makes cells at" in g.to_context()


# -------------------------------------------------------------------- queries

def test_multi_hop_path_found(g):
    chain(g)
    paths = g.find_paths("Acme Corp", "May recall")
    assert [[c.id for c in p] for p in paths] == [["c1", "c2", "c3"]]


def test_path_matching_is_case_insensitive(g):
    chain(g)
    assert g.find_paths("acme corp", "MAY RECALL")


def test_superseded_claims_are_not_traversed(g):
    _, c2, _ = chain(g)
    g.revise(c2.id, "model", "wrong plant", obj="Porto plant")
    assert g.find_paths("Acme Corp", "May recall") == []


def test_path_can_be_limited_to_agreed_claims(g):
    c1, c2, c3 = chain(g)
    for c in (c1, c3):
        g.respond(c.id, "nina", "agree")
    assert g.find_paths("Acme Corp", "May recall", statuses=[Status.ACCEPTED]) == []
    g.respond(c2.id, "nina", "agree")
    assert len(g.find_paths("Acme Corp", "May recall", statuses=[Status.ACCEPTED])) == 1


def test_max_hops_respected(g):
    chain(g)
    assert g.find_paths("Acme Corp", "May recall", max_hops=2) == []


def test_cycles_do_not_loop_forever(g):
    g.propose("model", "A", "r", "B", "x")
    g.propose("model", "B", "r", "A", "x")
    g.propose("model", "B", "r", "C", "x")
    paths = g.find_paths("A", "C", max_hops=10)
    assert [[c.object for c in p] for p in paths] == [["B", "C"]]


def test_equal_length_chains_ranked_by_support(g):
    c1, c2, c3 = chain(g)
    alt = g.propose("model", "VoltCo", "ships from", "Lyon plant", "doc 4", ["doc4"])
    for c in (c1, c3, alt):
        g.respond(c.id, "nina", "agree")
    g.respond(c2.id, "nina", "unsure", "which plant?")
    paths = g.find_paths("Acme Corp", "May recall")
    assert [p[1].id for p in paths] == [alt.id, c2.id]


def test_explain_path_names_weakest_link(g):
    c1, c2, c3 = chain(g)
    g.respond(c1.id, "nina", "agree")
    g.respond(c2.id, "nina", "disagree", "Porto, not Lyon")
    g.respond(c3.id, "nina", "agree")
    text = g.explain_path(g.find_paths("Acme Corp", "May recall")[0])
    assert "Weakest link: c2 (disputed)" in text
    assert "Porto, not Lyon" in text


def test_review_queue_most_contested_first(g):
    c1, c2, c3 = chain(g)
    g.respond(c1.id, "nina", "agree")
    g.respond(c2.id, "nina", "unsure", "which plant?")
    g.respond(c3.id, "nina", "disagree", "recall was in June")
    assert [c.id for c in g.review_queue()] == ["c3", "c2"]


def test_review_queue_puts_what_the_focus_depends_on_first(g):
    chain(g)
    side = g.propose("model", "VoltCo", "was founded in", "1998", "doc 2", ["doc2"])
    g.respond(side.id, "nina", "unsure", "could be 1999")  # more contested, but off the path
    focus = [("Acme Corp", "May recall")]
    assert [c.id for c in g.review_queue(focus=focus)] == ["c1", "c2", "c3", side.id]
    assert [c.id for c in g.review_queue()] == [side.id, "c1", "c2", "c3"]
    assert len(g.review_queue(focus=focus, limit=2)) == 2


# --------------------------------------------------------------------- export

def test_mermaid_distinguishes_agreed_from_open(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "nina", "agree")
    diagram = g.to_mermaid()
    assert diagram.startswith("graph LR")
    assert '-- "buys from" -->' in diagram
    assert '-. "makes cells at (proposed)" .->' in diagram


def test_mermaid_shows_qualifiers(g):
    g.propose("model", "A", "rel", "B", "because", qualifiers={"from": "2019"})
    assert '-. "rel (from: 2019) (proposed)" .->' in g.to_mermaid()


def test_context_lists_objections_questions_and_links(g):
    c1, c2, _ = chain(g)
    g.respond(c1.id, "nina", "disagree", "switched suppliers")
    q = g.ask("nina", "which plant?", about=[c2.id])
    c4 = g.revise(c2.id, "model", "renamed", obj="Lyon Plant")
    r = g.refine(c4.id, "model", "narrowing", qualifiers={"for customer": "Acme Corp"})
    g.answer(q.id, "model", r.id)
    context = g.to_context()
    assert "nina disagree: switched suppliers" in context
    assert "q1 (nina, about c4; open): which plant?" in context  # follows the revision
    assert "| refines c4\n" in context
    g.respond(r.id, "nina", "agree")
    context = g.to_context()
    assert f"q1 (nina, about c4; answered by {r.id}): which plant?" in context
    assert "| refines c4; answers q1\n" in context


def test_log_records_every_action(g):
    c1, c2, c3 = chain(g)
    g.respond(c1.id, "nina", "agree")
    g.revise(c2.id, "model", "narrowing", relation="ships from")
    q = g.ask("nina", "when?", about=[c3.id])
    r = g.refine(c3.id, "model", "dates", qualifiers={"in": "May 2026"})
    g.answer(q.id, "model", r.id)
    actions = [e["action"] for e in g.log]
    assert actions == ["add source"] * 4 + [
        "propose", "propose", "propose", "agree", "propose", "revise",
        "ask", "propose", "refine", "answer",
    ]
    assert [e["target"] for e in g.log[-4:]] == ["q1", "c5", "c3", "q1"]


def test_log_keeps_cited_sources_when_a_position_changes(g):
    c = g.propose("nina", "A", "rel", "B", "because")
    g.respond(c.id, "model", "agree", "doc 1 confirms it", sources=["doc1"])
    g.respond(c.id, "model", "disagree", "on reflection, no")
    assert c.positions["model"].sources == []
    assert g.log[-2]["detail"] == "doc 1 confirms it [sources: doc1]"


def test_exports_do_not_share_state_with_the_graph(g):
    c1, c2, _ = chain(g)
    q = g.ask("nina", "which plant?", about=[c2.id])
    data = g.to_dict()
    data["claims"][0]["sources"].append("made-up")
    data["log"].clear()
    assert c1.sources == ["doc1"] and g.log
    loaded = CommonGround.from_dict(data)
    loaded.answer(q.id, "model", c1.id)
    assert data["questions"][0]["answers"] == []
    assert g.questions[q.id].answers == []


def test_save_and_load_round_trip(g, tmp_path):
    g.add_participant("model-2", "model", identity="model")
    c1, c2, c3 = chain(g)
    g.respond(c1.id, "nina", "disagree", "switched suppliers")
    g.revise(c2.id, "model", "narrowing", relation="ships from")
    q = g.ask("nina", "when?", about=[c3.id])
    r = g.refine(c3.id, "model", "dates", qualifiers={"in": "May 2026"}, sources=["doc4"])
    g.answer(q.id, "model", r.id)
    g.respond(r.id, "model-2", "agree", "doc 4 dates it", sources=["doc4"])
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = CommonGround.load(path)
    assert loaded.to_dict() == g.to_dict()
    assert loaded.claims["c1"].status is Status.DISPUTED
    assert loaded.claims["c2"].status is Status.SUPERSEDED
    assert loaded.claims[r.id].status is Status.PROPOSED  # same identity as the proposer
    assert loaded.open_questions()[0].id == q.id
    # a loaded graph keeps numbering claims and questions correctly
    assert loaded.propose("nina", "X", "r", "Y", "because").id == "c6"
    assert loaded.ask("nina", "and?").id == "q2"


def test_settings_survive_save_and_load(tmp_path):
    g = CommonGround(vocabulary=["rel"], agreement_reason_required=())
    g.add_participant("model", "model")
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = CommonGround.load(path)
    assert loaded.vocabulary == ["rel"]
    assert loaded.agreement_reason_required == frozenset()
