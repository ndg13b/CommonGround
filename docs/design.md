# Design notes

These notes explain the choices behind Common Ground, the problems they are meant to solve, and what is still unsolved. The [README](../README.md) has the short version. Version 0.2 came out of a review of 0.1, and each section below says what changed and why.

## The graph is the common ground, not the conversation

In Clark & Brennan's (1991) account, people communicate by *grounding*: building up mutual knowledge, checking it as they go, and repairing it when one side turns out to have misunderstood. The mutual knowledge they build is their *common ground*. The conversation is how they build it.

Common Ground keeps that split. People will not want to talk to a model in triples, and they shouldn't have to: the conversation can stay in plain language. What the graph adds is a place where the results of the conversation are kept. Each turn can propose, challenge, narrow or question a claim, and both sides can then point at that claim by its id instead of paraphrasing something said forty messages earlier. In a chat transcript, what has been agreed is implicit and scattered. In the graph it is explicit, addressable, and comes with its reasons.

In practice:

- `to_context()` renders the common ground so far as text a model can read at the start of each turn: claims with their objections, which claim refines which and which questions each answers; every question and whether it is answered; the sources that can be cited; and the relations in use.
- A model's reply can include actions on the graph: propose, respond, ask, answer, refine, revise. Turning a model's reply into those actions is the job of the LLM adapter on the roadmap.
- People take the same actions through whatever interface they use; each one is an ordinary method call.

## Doubts and questions are different

Version 0.1 had one way to flag a problem with a claim: say you were `unsure` about it. Its own worked example used that for something that wasn't a doubt. Nina did not doubt that VoltCo makes cells in Lyon; she wanted to know whether that was where *Acme's* cells came from. Marking the claim uncertain misdescribed it, and the fix that followed (revising the claim) removed a true fact from the graph.

Version 0.2 separates the cases, following IBIS (Issue-Based Information System; Kunz & Rittel, 1970), which treats *issues* (questions) separately from the *positions* taken on them and the *arguments* for and against:

| You think the claim… | Do this | Effect on the claim |
|---|---|---|
| is false | `respond(..., "disagree", reason)` | becomes `disputed` |
| may be false | `respond(..., "unsure", reason)` | becomes `uncertain` |
| is true but doesn't say enough | `ask(..., about=[claim_id])` | none; every chain through it, or through a revision or refinement of it, reports the question until a claim in that chain answers it |

A question is *answered* when someone offers a claim as the answer with `answer()`, and someone other than the person who offered it then agrees with that claim, leaving it accepted. Agreement given before the offer doesn't count, so nobody can close a question with a claim that was already settled, and a question can't be answered by the claim it is about. Nobody has to take an answer on trust.

The same split applies to changing a claim:

- `revise()` replaces a claim that was **wrong**. The original is marked `superseded` and leaves the active graph, but stays on record. Questions asked about it carry over to its replacement.
- `refine()` adds a narrower version of a claim that was **true but too broad**. Both stay active and each links to the other, including after the refinement is itself revised. Questions asked about the original apply to the refinement too, so narrowing a claim can't route around an open question.

## Agreement has to mean something

A claim's status is only as informative as the agreement behind it, and three things in 0.1 made agreement cheap.

**Unexplained agreement.** Objections needed reasons; agreement didn't. For people that is usually fine: a quick "yes" from someone who knows the domain carries information. For language models it is not. Models tend to agree with the person they are talking to, and to give way when challenged, whether or not they are right (Sharma et al., 2023), so an unexplained "agree" from a model says little. In 0.2, participants of the kinds listed in `agreement_reason_required` (models, by default) must give a reason when they agree, and can cite sources for it. `CommonGround(agreement_reason_required=("human", "model"))` holds everyone to it; `()` holds nobody to it.

This does not stop a model inventing a reason. It puts the reason on record, where it can be checked and challenged.

**Uncheckable evidence.** Sources were free text, so a claim could cite a document nobody had provided. In 0.2, claims and responses can only cite sources registered with `add_source()`, and the log records who supplied each one: the system (for example, the documents a model was given) or a named participant. A model can still register a source, but then the graph shows that the model is the only one vouching for it. The log also keeps the sources each response cited, so changing a response doesn't erase the evidence behind the old one.

**Self-agreement.** "Accepted" required agreement from someone other than the proposer, checked by name, so two instances of the same model under different names could accept each other's claims. In 0.2 each participant has an `identity` (by default, its name). Participants that share an identity count as one voice, and agreement only counts when it comes from a different identity than the proposer's. Registering a name again with a different kind or identity is refused, so a participant can't change identity partway through a graph.

## Claims carry their conditions

Triples lose detail. Real claims hold for a customer, a period or a product line, and in 0.1 the only place to put that was the relation name. The 0.1 example did exactly that with "ships Acme's orders from", a relation unlikely to match anything else in the graph.

- **Qualifiers**, after Wikidata's, are key-value pairs on a claim: `{"for customer": "Acme Corp"}`, `{"from": "2019", "until": "2023"}`. They are shown wherever the claim is, and a refinement adds to them.
- **A relation vocabulary.** Free-text relations drift ("makes cells at", "produces cells in") and the graph splits into near-duplicates. `to_context()` lists the relations already in use so a model can reuse them, and `CommonGround(vocabulary=[...])` turns that into a hard rule.

Still missing: entity resolution (merging "VoltCo" and "VoltCo SA") and queries that understand time qualifiers.

## Human attention is the scarce resource

A model can propose a thousand claims in a minute; a person can check a few dozen. Which claims a person sees decides whether the graph stays trustworthy. `review_queue(focus=...)` takes the questions currently being asked, as start and end entities, and puts first the unsettled claims that most of them depend on, then the most contested. Claims that no current question rests on wait. `find_paths()` does the same for answers: among chains of equal length, the best-supported comes first.

## Known limits

- **A chain is not a proof.** `find_paths()` finds chains of claims, not valid inferences. "Acme buys from VoltCo" and "VoltCo makes cells at Lyon" form a chain but don't establish that Acme's cells come from Lyon. Questions are the current way to flag this; nothing yet checks whether relations compose.
- **Simple status rules.** One objection from anyone makes a claim `disputed`, and one independent agreement makes it `accepted`. With many participants these rules will need weighting or a quorum.
- **Everyone's word weighs the same.** The log records who said what, but statuses take no account of expertise or track record.

## How to test the idea

Everything depends on how a live model behaves under these rules, so the next step is the LLM adapter, paired with a small study.

- **Task.** Participants answer multi-hop questions over a small set of documents, with errors planted at known points: a wrong link, a missing condition, a claim that is true in general but not for the case asked about.
- **Conditions.** (a) Chat with a model that has the documents. (b) The same chat, plus the graph: the model's claims appear as a graph the person can agree with, challenge and question, and the model reads `to_context()` each turn.
- **Measures.** How many planted errors people catch; whether their confidence in their final answer matches whether it is right; how often the model gives way on a correct claim when challenged; and time taken.

If the graph helps people catch errors that chat lets through, without costing much time, that is the result the idea needs.

## Where this sits

Most of the machinery is older than language models:

- **Truth maintenance systems** (Doyle, 1979) record a justification for every belief and withdraw whatever depends on a belief that is withdrawn. Common Ground's rationales and supersession are a lightweight version, without the automatic withdrawal.
- **IBIS** (Kunz & Rittel, 1970) structures discussion as issues, positions and arguments. Common Ground's questions are IBIS issues, its claims are positions, and the reasons attached to them are arguments.
- **Abstract argumentation** (Dung, 1995) defines which sets of arguments can stand together, given which arguments attack which. Common Ground's statuses are a much simpler, per-claim version.
- **Wikidata** (Vrandečić & Krötzsch, 2014) puts references on every statement and has qualifiers, statement ranks and talk pages: provenance and disagreement, handled by a community at scale.

What is new is using that machinery as a two-way channel between people and language models, in which both sides propose, both must justify, and the model's agreement is held to account.

## References

- Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In L. B. Resnick, J. M. Levine & S. D. Teasley (Eds.), *Perspectives on Socially Shared Cognition* (pp. 127–149). American Psychological Association.
- Doyle, J. (1979). A truth maintenance system. *Artificial Intelligence*, 12(3), 231–272.
- Dung, P. M. (1995). On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. *Artificial Intelligence*, 77(2), 321–357.
- Kunz, W., & Rittel, H. W. J. (1970). *Issues as Elements of Information Systems* (Working Paper No. 131). Institute of Urban and Regional Development, University of California, Berkeley.
- Sharma, M., et al. (2023). Towards understanding sycophancy in language models. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548).
- Vrandečić, D., & Krötzsch, M. (2014). Wikidata: a free collaborative knowledgebase. *Communications of the ACM*, 57(10), 78–85.
