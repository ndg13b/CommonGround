import pytest

from commonground import CommonGround, Stance, Status


@pytest.fixture
def g() -> CommonGround:
    graph = CommonGround(clock=lambda: "2026-01-01T00:00:00+00:00")
    graph.add_participant("model", "model")
    graph.add_participant("nina", "human")
    return graph


def chain(g: CommonGround):
    c1 = g.propose("model", "Acme Corp", "buys from", "VoltCo", "doc 1 says so", ["doc1"])
    c2 = g.propose("model", "VoltCo", "makes cells at", "Lyon plant", "doc 2 says so", ["doc2"])
    c3 = g.propose("model", "Lyon plant", "had", "May recall", "doc 3 says so", ["doc3"])
    return c1, c2, c3


# ---------------------------------------------------------------- claim rules

def test_new_claim_is_proposed_until_someone_else_agrees(g):
    c1, *_ = chain(g)
    assert c1.status is Status.PROPOSED
    g.respond(c1.id, "nina", "agree")
    assert c1.status is Status.ACCEPTED


def test_proposer_agreeing_again_does_not_accept_own_claim(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "model", "agree")
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


def test_cannot_respond_to_or_revise_superseded_claim(g):
    _, c2, _ = chain(g)
    g.revise(c2.id, "model", "narrowing", relation="ships from")
    with pytest.raises(ValueError):
        g.respond(c2.id, "nina", "agree")
    with pytest.raises(ValueError):
        g.revise(c2.id, "model", "again", relation="other")


def test_revision_requires_reason(g):
    _, c2, _ = chain(g)
    with pytest.raises(ValueError):
        g.revise(c2.id, "model", "", relation="x")


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


def test_explain_path_names_weakest_link(g):
    c1, c2, c3 = chain(g)
    g.respond(c1.id, "nina", "agree")
    g.respond(c2.id, "nina", "disagree", "Porto, not Lyon")
    g.respond(c3.id, "nina", "agree")
    text = g.explain_path(g.find_paths("Acme Corp", "May recall")[0])
    assert "Weakest link: c2 (disputed)" in text
    assert "Porto, not Lyon" in text


def test_open_questions_sorted_most_contested_first(g):
    c1, c2, c3 = chain(g)
    g.respond(c1.id, "nina", "agree")
    g.respond(c2.id, "nina", "unsure", "which plant?")
    g.respond(c3.id, "nina", "disagree", "recall was in June")
    assert [c.id for c in g.open_questions()] == ["c3", "c2"]


# --------------------------------------------------------------------- export

def test_mermaid_distinguishes_agreed_from_open(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "nina", "agree")
    diagram = g.to_mermaid()
    assert diagram.startswith("graph LR")
    assert '-- "buys from" -->' in diagram
    assert '-. "makes cells at (proposed)" .->' in diagram


def test_context_lists_objections(g):
    c1, *_ = chain(g)
    g.respond(c1.id, "nina", "disagree", "switched suppliers")
    assert "nina disagree: switched suppliers" in g.to_context()


def test_log_records_every_action(g):
    c1, c2, _ = chain(g)
    g.respond(c1.id, "nina", "agree")
    g.revise(c2.id, "model", "narrowing", relation="ships from")
    actions = [e["action"] for e in g.log]
    assert actions == ["propose", "propose", "propose", "agree", "propose", "revise"]


def test_save_and_load_round_trip(g, tmp_path):
    c1, c2, _ = chain(g)
    g.respond(c1.id, "nina", "disagree", "switched suppliers")
    g.revise(c2.id, "model", "narrowing", relation="ships from")
    path = tmp_path / "graph.json"
    g.save(path)
    loaded = CommonGround.load(path)
    assert loaded.to_dict() == g.to_dict()
    assert loaded.claims["c1"].status is Status.DISPUTED
    assert loaded.claims["c2"].status is Status.SUPERSEDED
    # a loaded graph keeps numbering claims correctly
    assert loaded.propose("nina", "X", "r", "Y", "because").id == "c5"
