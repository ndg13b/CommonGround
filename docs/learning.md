# Learning from the graph

The rest of Common Ground uses the graph as common ground between people and a model. These notes go further: could a model use a graph like this to *learn*, changing its own weights in response to knowledge that has been checked, rather than only reading more context? The first pieces are built ([what exists today](#what-exists-today)); most of what follows is design.

## Where this starts

A language model is trained once, and training sets its weights. After that the weights are fixed. Everything in a conversation, including any "memory" feature a product adds, is text the fixed model reads. A model does adapt to what it reads within a conversation (in-context learning), but that adaptation disappears with the context. Models change only between versions, when they are retrained offline on curated data and tested.

The question is whether a model could keep a knowledge graph of what it knows, notice when new information challenges it, check that information, and, if a change is warranted, retrain so that the change reaches its weights.

That has the shape of *complementary learning systems* theory in neuroscience (McClelland, McNaughton & O'Reilly, 1995): the hippocampus learns quickly, and what it holds is gradually consolidated into the neocortex, partly during sleep. Here the graph is the fast store and training is the slow consolidation.

## Why it is hard

- **Forgetting.** Training on new material tends to overwrite old knowledge, because the network doesn't know what to protect ("catastrophic forgetting").
- **Superposition.** Networks pack many unrelated concepts into the same neurons (Elhage et al., 2022), so changing one fact can disturb others.
- **New facts don't spread.** Editing one fact in a model often leaves what follows from it unchanged (Cohen et al., 2023). Training on "A is B" doesn't teach "B is A" (Berglund et al., 2023), and fine-tuning on new facts can make a model hallucinate more (Gekhman et al., 2024).
- **Poisoning.** If what people say can change the weights, anyone can try to teach the model falsehoods.
- **Self-checking isn't independent.** Checkers created from the same model share its blind spots.
- **Safety drift.** Fine-tuning, even on harmless data, can wear away a model's safety training (Qi et al., 2023).
- **Unlearning is hard.** Removing knowledge from weights is an open problem, and hidden behaviours have survived safety training (Hubinger et al., 2024).
- **Scale.** One model serves many people. Whose corrections win?

## The design

### 1. The graph is the source of truth; the weights are a rebuildable copy

This is the most important principle. Errors are fixed in the graph, where they can be superseded and audited. The weights are then retrained, or a module swapped, from the corrected graph, rather than edited in place.

To make that affordable, knowledge should live in separable modules, such as small add-on weights (LoRA, low-rank adaptation, adapters) per level or per topic, which can be tested, replaced or deleted. Methods for keeping knowledge apart exist: parameter isolation (for example Rusu et al., 2016) and gradient routing (Cloud et al., 2024). The brain does something similar: the hippocampus stores similar memories in deliberately different patterns ("pattern separation"). But superposition means the separation is never clean, and too much of it loses the sharing that lets a model generalise.

### 2. Three levels, ordered by entrenchment

- **Principles:** purpose and ethics. Only people may propose or change them.
- **Concepts:** general understanding that facts are read through.
- **Facts:** episodes and specific facts. They sit at the edge and are revised first.

This follows Quine's "web of belief" and *epistemic entrenchment* in belief-revision theory (Gärdenfors & Makinson, 1988): when evidence conflicts, revise at the edge first, and give up entrenched beliefs last. It also mirrors the split in psychology between episodic and semantic memory. Entrenched claims can still change, because a core that can never change can never be corrected. They just need more to change.

### 3. Every change has a branch

A claim can say what it rests on, such as the facts and concepts a conclusion was drawn from. When a claim is revised, disputed or doubted, everything downstream of it is flagged for re-checking, with the reason. A flagged claim is then either re-affirmed or disputed. Re-affirming means revising it onto the current versions of what it rests on, after which it needs fresh agreement. Support can't be circular: a revision that would end up resting on itself is refused, because circular support could never settle.

This is the core of truth maintenance systems (Doyle, 1979), without their automatic withdrawal: a person or a model decides. It also answers the ripple problem from [Why it is hard](#why-it-is-hard). The graph knows what should change when one fact does, even when a model's weights don't.

### 4. A tunable bar for what reaches the weights

*Accepted* means the participants agree. *Ready* means a claim may be trained into the weights. Readiness is a higher bar, set per level by three numbers:

- how many distinct identities must agree, counting the proposer;
- how many of them must be human;
- how many independent sources the claim must cite. Copies of one original count once; a claim that rests on other claims gets its evidence from them instead.

Whatever the level, a ready claim also has no open question on it and needs no re-checking, and everything it rests on is ready too. By default, facts need an outside source, and concepts and principles need a person among the voices agreeing to them. The bar should rise with the stakes. One part of this is not built yet: raising the bar with the number of claims that depend on a claim.

This is the defence against poisoning, and it is only as good as the sources. An attacker can create sources, so tracking where a source came from matters as much as counting sources.

### 5. Tests generated from the graph

Before and after every training run, the model is given true/false statements generated from the graph:

- **core:** principles and concepts. These must pass every time; an update that breaks one is rejected.
- **fact:** the claims being trained in.
- **negative:** what was revised away, which the model should now judge false.
- **ripple:** claims resting on something that just changed, which check that the change reached them.

This turns the graph into a regression test suite for the model: tests that catch an update which breaks what already worked. True/false statements are a shallow test; multi-step questions generated from chains of claims are the natural next step.

### 6. Principles come first, and stay correctable

Asimov's Three Laws are the obvious reference, and his stories are largely about how fixed rules fail. In "Runaround", a robot circles endlessly because two laws balance each other out. Rules need interpretation, and the interpretation lives in the same weights that training changes. So:

- principles are text given to the model on every run (`to_context()` lists them first, as taking priority), so training can't erase them;
- passing the core probes is a condition of every update;
- only people change principles, deliberately, never the self-updating loop.

Real-world counterparts include Constitutional AI (Bai et al., 2022), which trains a model on a written set of principles, and deliberative alignment (Guan et al., 2024), which trains a model to reason over a safety specification.

One caveat, studied under the name *corrigibility* (Soares et al., 2015): a system whose top layer can't be changed may also resist correction when that layer is wrong. So the top layer should include accepting legitimate human oversight and correction. Protecting values well also means protecting people's ability to fix them.

### 7. Checking against the outside world

Checking agents should consult sources outside the model. The strongest checks need no judgement at all: run the code, query the database, do the calculation. Checkers built on different models reduce shared blind spots, and changes to entrenched claims go to a person. Common Ground's identity rule already captures part of this: instances of the same model count as one voice.

## Is this recursive self-improvement?

Not quite. Recursive self-improvement, I. J. Good's (1965) "intelligence explosion", is a system improving its *ability to improve itself*, so that the gains compound. This loop improves what a model *knows*, not how well it learns or reasons.

It would move toward recursive self-improvement if it started training on the model's own reasoning to raise its skills, or changing its own training process. It already shares that idea's central governance problem, though: a system deciding what to change about itself. That is why frontier AI labs' safety frameworks watch autonomous AI research and development closely, and why the people in this loop matter.

## What exists today

| Idea | In Common Ground |
|---|---|
| Levels | `propose(..., level="principle" \| "concept" \| "fact")`. Only humans may propose, revise or refine principles. |
| What a claim rests on | `depends_on=[...]` on `propose()`, `revise()` and `refine()`; `dependents(claim_id)` lists everything downstream. |
| Branch re-checking | `needs_recheck()`, most entrenched first, with reasons. Also shown by `why()`, `explain_path()` and `to_context()`. |
| Independent sources | `add_source(..., origin=...)` records what a source repeats; `origin_of()` finds the original. |
| The bar for the weights | `plan_consolidation(graph, thresholds)` returns the claims that are ready and those held back, with reasons. |
| Tests from the graph | `plan.probes`: core, fact, negative and ripple probes. |
| Principles first | `to_context()` lists principles first, then concepts, then facts. |

[`examples/learning_loop.py`](../examples/learning_loop.py) walks through all of it: a news report is corrected, the branch resting on it is re-checked, and the new plan includes negative and ripple probes.

## Not built yet

- The LLM adapter from the main roadmap, so that a live model takes actions on the graph.
- Checking agents that consult outside sources.
- The training step: turning ready claims into training data for an adapter, running the probes before and after, and rejecting an update that fails a core probe.
- Separable modules per level or topic.
- A bar that rises with the number of claims depending on a claim.
- Multi-step question probes generated from chains of claims.

## A first experiment

- Take a small open model with a LoRA adapter, and a graph like the worked example, scaled up.
- Train the claims the plan marks ready into the adapter. Compare against the same model with `to_context()` in its prompt, and against the model alone.
- Measure:
  - accuracy on the fact and ripple probes;
  - the negative probes, to see whether old beliefs are gone;
  - the core probes before and after;
  - performance on an unrelated benchmark, to detect forgetting;
  - the cost of training.
- Then revise a fact, and compare retraining the adapter from the corrected graph with editing the model in place.

If consolidated knowledge spreads to the ripple probes better than in-context knowledge does, without forgetting elsewhere, the approach is worth scaling.

## References

Compiled from memory, without network access to check them: confirm titles and arXiv identifiers before citing.

- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073.
- Berglund, L., et al. (2023). The reversal curse: LLMs trained on "A is B" fail to learn "B is A". arXiv:2309.12288.
- Cloud, A., et al. (2024). Gradient routing: Masking gradients to localize computation in neural networks. arXiv:2410.04332.
- Cohen, R., et al. (2023). Evaluating the ripple effects of knowledge editing in language models. arXiv:2307.12976.
- Doyle, J. (1979). A truth maintenance system. *Artificial Intelligence*, 12(3), 231–272.
- Elhage, N., et al. (2022). Toy models of superposition. arXiv:2209.10652.
- Gärdenfors, P., & Makinson, D. (1988). Revisions of knowledge systems using epistemic entrenchment. *Proceedings of the 2nd Conference on Theoretical Aspects of Reasoning about Knowledge*, 83–95.
- Gekhman, Z., et al. (2024). Does fine-tuning LLMs on new knowledge encourage hallucinations? arXiv:2405.05904.
- Good, I. J. (1965). Speculations concerning the first ultraintelligent machine. *Advances in Computers*, 6, 31–88.
- Guan, M. Y., et al. (2024). Deliberative alignment: Reasoning enables safer language models. arXiv:2412.16339.
- Hubinger, E., et al. (2024). Sleeper agents: Training deceptive LLMs that persist through safety training. arXiv:2401.05566.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457.
- Qi, X., et al. (2023). Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv:2310.03693.
- Quine, W. V. O., & Ullian, J. S. (1970). *The Web of Belief*. Random House.
- Rusu, A. A., et al. (2016). Progressive neural networks. arXiv:1606.04671.
- Soares, N., Fallenstein, B., Yudkowsky, E., & Armstrong, S. (2015). Corrigibility. *AAAI Workshop on AI and Ethics*.
