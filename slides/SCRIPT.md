# Phase 6: voice-over script

**Topic 15, Observation Budgets and Padding Effects in Raw-Byte Network Traffic
Classification.** Muhammad Talha Arshad (i232548) and Zain Shahid (i232582),
BS Data Science, FAST-NUCES Islamabad.

Read against `Phase6_Presentation.pdf`. Slide numbers match the number printed
in the top right of each slide.

---

## How the talk is split

Both members present, and each presents a mix of conceptual and technical
material rather than one taking "the theory" and the other "the code". Each
member also presents results of their own.

| Member | Slides | Covers |
|---|---|---|
| **Talha** (i232548) | 1&ndash;5 | problem, the padding confound, research gap, objectives |
| **Zain** (i232582) | 6&ndash;11 | methodology, implementation, **live demonstration** |
| **Talha** (i232548) | 12&ndash;16 | results: leakage gate, budget response, architecture control, per-class, padding |
| **Zain** (i232582) | 17&ndash;19 | results: accuracy per byte, the split confound, comparison with base papers |
| **Talha** (i232548) | 20 | key findings |
| **Zain** (i232582) | 21 | limitations and threats to validity |
| **Talha** (i232548) | 22 | conclusion and future work |
| **Zain** (i232582) | 23 | closing, repository, questions |

**Speaking time is about 21 and a half minutes** at a normal narration pace of
150 words per minute, with the live demonstration budgeted at 2 minutes. The
script is 3,185 spoken words: 1,534 for Talha and 1,651 for Zain, so the two
shares are even to within three quarters of a minute.

If a shorter video is wanted, **this cut brings it to about 15 minutes** and
still covers all seven required topics, with both members still presenting
problem, method, results and conclusion:

| Cut | Saves |
|---|---|
| Slide 15 (per-class behaviour) entirely; keep one sentence about it inside slide 20 | 0:55 |
| Slide 18 (the split confound); its headline stays on slide 20 | 1:00 |
| Slide 21 (limitations); keep only the receptive-field threat, spoken over slide 22 | 1:05 |
| Demo: keep the causality test and the budget-sweep JSON, drop the encoder read-through and the report | 0:50 |
| Slides 3, 7, 8, 19: drop the final paragraph of each | 1:40 |

Do not cut slides 13, 17 or 20. Those three carry the saturation result, the
byte saving and the findings, which is what the project is actually for.

The deck is 25 pages: **slides 1 to 23 are the talk**, and pages 24 and 25 are
backup slides B1 and B2, carrying the ISCX budget response and the ISCX
frontier for questions. Stop advancing at slide 23 while recording.

Every number spoken below is on the slide it belongs to, and all of them come
from the result JSON in `results/`. Nothing is rounded differently between the
deck, the paper and this script.

---

# Part 1. Talha: the problem and what we set out to measure

## Slide 1, title

> Good afternoon. This is our Computer Networks research project. The topic is
> observation budgets and padding effects in raw-byte traffic classification,
> and we framed it as a question: how many bytes does encrypted traffic
> classification actually need?
>
> I am Muhammad Talha Arshad, i232548, presenting with Zain Shahid, i232582. I
> will take the problem, the gap and most of the results. Zain takes the
> methodology, the implementation and a live demonstration of the code.

## Slide 2, the problem

> A deep traffic classifier does not see a flow. It sees the first B bytes of
> it, zero-padded to a fixed length. That prefix is the observation budget.
>
> The budget is not a knob on the side; it is the operating point of the
> system. A 1024-byte budget cannot decide before 1024 bytes have arrived, so
> it sets detection latency. A middlebox holds B bytes for every concurrent
> flow, so it sets memory. Blocking, shaping and alerting all wait on the same
> prefix.
>
> So we looked at what the field does. Across twelve studies from 2025 and
> 2026, the budget runs from five to twenty packets, or 784 to 1024 bytes. In
> not one of them is it measured. Where a reason is given it is architectural,
> such as a byte count that reshapes into a square image, not empirical. Zero
> out of twelve report accuracy as a function of the budget. That curve is the
> object of this project.

## Slide 3, the padding confound

> It is worse than an unexamined hyperparameter, and this is the point I most
> want to land.
>
> Flows are short and heavy-tailed. The median ISCX session is 274 bytes, the
> median USTC session 474. So a budget larger than that does not read more
> traffic, it reads more padding. At a 512-byte budget, the average ISCX
> session is 48 percent padding; at 1500 bytes it is 70 percent.
>
> That means raising the budget always raises the padding fraction. Any
> reported gain from a larger budget mixes two effects: more real bytes, and
> more padding the model can use as a length cue. Nobody separates them. A
> paper concluding that 1024 bytes works best may have measured a length
> classifier and called it a payload classifier.

## Slide 4, the research gap

> Four gaps come out of the literature, and we address them in this order.
>
> Gap one, the budget is inherited rather than measured. Gap two, aggregate
> accuracy conceals per-class failure, and I will show you a concrete case. Gap
> three, corpora are called encrypted without anyone measuring how much of them
> is. Gap four, shortcut learning from IP addresses and ports is rarely
> excluded, and one of our base papers shows that leaving those bytes in is
> worth up to 0.45 accuracy.
>
> Together these mean published budgets cannot be compared, and cannot be used
> as evidence about how much of a flow a classifier needs.

## Slide 5, objectives

> Five objectives, each with a measurable answer.
>
> One, measure the budget response on two corpora and locate the saturation
> budget: the smallest budget beyond which no larger budget is reliably better.
> Two, test whether that answer is a property of the model, by repeating the
> sweep with a structurally different architecture. Three, decompose the
> padding effect into a representation term and a padding-symbol term. Four,
> build a classifier that reads only what it needs, with a causality guarantee
> so that bytes read is a real cost. Five, report honestly: exclude shortcut
> learning, report per-class behaviour, and quantify how much of a published
> number the train-test split explains on its own.
>
> Two public corpora, 385 thousand sessions, 31 classes, three to eight seeds
> behind every number. Zain will take you through how we built it.

---

# Part 2. Zain: methodology, implementation and the demonstration

## Slide 6, the pipeline

> Thank you Talha. I am Zain Shahid, i232582.
>
> The design requirement follows from the objective. If the budget is to be a
> measured variable, it must be the only thing that changes between two points
> on the curve.
>
> So one preprocessing pass writes a fixed 2048-byte HDF5 session store, and
> every budget is a slice of that one store. Parsing, sessionization,
> deduplication and anonymisation happen once, before any budget exists, so
> class balance and session boundaries cannot drift between 256 and 512 bytes.
> In the diagram, only the dashed region repeats per budget.
>
> Splits are group-wise by capture file wherever the corpus allows, so no
> capture appears on both sides of the boundary. Every point is three seeds,
> reported as a mean and a standard deviation.

## Slide 7, the leakage gate

> Gap four is usually handled with one sentence promising identifiers were
> removed. We made it a precondition the pipeline enforces.
>
> Three probes train on byte positions alone. The gate is the condition on
> screen: a probe trained only on the anonymised identifier bytes must score no
> higher than the majority-class rate plus a margin of 0.1. If it does, the
> anonymisation has failed and the pipeline halts. It does not warn, it stops.
>
> The probes do different jobs. `id_positions` is the gate. `ident` measures
> shortcut strength by training on those fields before anonymisation, to find
> out what the shortcut would have been worth. And `ports` masks the port bytes
> specifically, separating port memorisation from payload structure.

## Slide 8, padding decomposed

> For the padding question we built five input modes.
>
> `zero` is the baseline, each byte as a scalar with padding as zero. `embed`
> swaps in a learned embedding per byte value but leaves padding as zero.
> `mask` keeps the embedding and adds a distinct PAD symbol. `convmask` also
> makes the convolution ignore padded positions. And `length_balanced`
> resamples classes to equal length distributions, removing length as a cue.
>
> The literature changes representation and padding symbol together, then
> credits the whole gain to padding-aware input. Comparing zero to embed to
> mask separates the two terms, and every term is compared against the seed
> spread, so anything smaller than seed noise is reported as inconclusive
> rather than as a finding.

## Slide 9, the causal early-exit classifier

> This is where the project could most easily have fooled itself.
>
> The idea is simple: heads at 64, 128, 256, 512 and 1024 bytes; at inference,
> stop at the first head whose confidence clears a threshold tau, and charge
> that flow only the bytes the head consumed.
>
> But the claim only holds if head k is incapable of having seen byte k plus
> one. Otherwise the saving is nominal and the frontier is a fiction.
>
> So the encoder is six causal dilated convolution blocks, dilations one
> through 32, kernel five, 96 channels, left-padding only, receptive field 505
> bytes. The normalisation is per-position, `ChannelNorm1d`, never BatchNorm or
> GroupNorm, because both average over the length axis, and that one detail
> would let a byte at position 900 shift the activation at position 10.
>
> We prove that as a proposition by induction over the layer index, and we also
> test it. The test perturbs a byte and measures how far forward the change can
> reach: 505 bytes forward, matching the analysis exactly, and zero backwards.
> It is in the regression set because a one-line change of normalisation layer
> would destroy the property while every accuracy number stayed plausible.

## Slide 10, implementation

> Briefly, what was built. Fifteen modules: a streaming pcap and pcapng reader,
> five-tuple sessionization and the HDF5 store, the slicer that applies budgets
> and padding modes, four model families including the early-exit network, and
> the experiment layer producing E1 through E4 and four ablations. 385 thousand
> sessions out of about 55 gigabytes of captures, into 483 megabytes of stores.
>
> Reproducibility here is mechanical, not aspirational. Every table and figure
> is generated from the result JSON by `make_report.py`, so no number is typed
> by hand in the paper, the poster or this deck. `study.sh` is stage-resumable,
> so a run that dies at epoch 30 resumes rather than restarts. And five
> regression tests, including one that plants a cue at a known byte depth and
> requires the sweep to find it.
>
> Let me show you some of this running.

## Slide 11, live demonstration

> **[Switch to the terminal. Follow `DEMO_RUNSHEET.md`. Narrate as the commands
> run, roughly as below.]**
>
> > First the receptive-field probe, because the whole early-exit result
> > depends on it. It changes one byte at position 100 and measures how far
> > that change travels. Positions 100 to 604 move, so the forward reach is 505
> > bytes, exactly the analytic figure, and the backward reach is zero. The
> > encoder cannot see ahead.
>
> > Now the causality test proper. The important block is the second one: head
> > k fed only its own prefix produces logits identical to head k fed the whole
> > sequence, to zero difference. That is what makes bytes read honest.
>
> > Here is the encoder, `CausalBlock` in `models.py`. The line the guarantee
> > rests on is this one, `ChannelNorm1d`. Swap it for BatchNorm and the
> > accuracy numbers barely move, but the saving becomes a fiction.
>
> > Now the experiments. This is the USTC budget sweep, straight out of its
> > JSON, no retyping. Macro-F1 is 0.9731 at 512 bytes, 0.9564 at 784, 0.9367
> > at 1500. The curve goes down, and Talha will come back to that.
>
> > And the same comparison in the significance file. McNemar's exact test, 512
> > against 784: n-one-zero is 2488, sessions correct at 512 and wrong at 784,
> > against 94 going the other way, p below ten to the minus four.
>
> > Finally the generated report. Every table here is produced from those JSON
> > files by `make_report.py`, which is why we can say nothing in the deck was
> > typed by hand.
>
> **[Switch back to the slides. Hand back to Talha.]**

---

# Part 3. Talha: results

## Slide 12, the leakage gate result

> Thanks Zain. Starting with the gate, because nothing else means anything
> without it.
>
> On USTC a probe trained on the anonymised identifier positions reaches 0.1351
> accuracy against a majority-class rate of 0.1359, so 0.0008 below chance. On
> ISCX it is 0.1095 against 0.2258. The identifier bytes are provably
> unlearnable, the gate opens, and the sweep may run.
>
> Now the other side. The `ident` probe, trained on those fields before
> anonymisation, reaches 0.53 accuracy on ISCX and 0.39 on USTC. So on ISCX,
> more than half the label is recoverable from addresses and ports with no
> payload at all. The shortcut is not hypothetical. That is Gap four, measured,
> and it is why everything that follows is about payload bytes rather than
> about which host talked to which.

## Slide 13, the budget response

> This is the central result. Macro-F1 against byte budget on USTC, three
> seeds, shaded band is the seed spread.
>
> Saturation falls at 512 bytes. On ISCX it falls at 256. The vertical markers
> are the budgets used by the twelve reviewed studies, and every one of them
> sits at or past saturation.
>
> But look past it. Macro-F1 goes 0.9731 at 512 bytes, 0.9564 at 784, 0.9540 at
> 1024, 0.9367 at 1500. It falls. The obvious objection is seed noise, so we
> tested it: McNemar's exact test on the paired 512 against 784 comparison
> finds 2488 sessions correct at 512 and wrong at 784, against 94 the other
> way, p below ten to the minus four.
>
> So reading more of the flow is not the safe choice people treat it as. Past
> saturation it is measurably worse, because the extra bytes are mostly
> padding.

## Slide 14, the architecture control

> There is an obvious objection, and we take it seriously: maybe a 2017
> convolutional network simply cannot use bytes past 512, and a modern
> architecture would.
>
> So we replaced the whole convolutional stack with a byte transformer,
> self-attention over byte embeddings, on the same store, split and seeds. Only
> the model changed.
>
> The two families are not separable. The difference at peak is 0.0050
> macro-F1 against a seed spread of 0.0379, and both saturate at two packets.
>
> That is what makes the finding generalise. Saturation is a property of where
> the discriminative bytes lie in the flow, not of the network reading them.

## Slide 15, per-class behaviour

> Gap two was that the aggregate conceals per-class failure. Here is the
> concrete form.
>
> ISCX at 784 bytes, uncapped: accuracy 0.4013, macro-F1 0.5110, weighted-F1
> 0.3798. Two failure modes are invisible in those numbers. `vpn_chat` is the
> second-largest class in the test set, 686 sessions, and gets a recall of
> 0.042. It almost never predicts the class, though when it does it is usually
> right, precision 0.744. Its mirror image is `vpn_file`, recall 1.000 at
> precision 0.241, acting as a sink for every session the model cannot place.
>
> And these are not thin classes. The 44-session `vpn_email` class reaches
> 0.955 recall, so a support-weighted summary would predict the opposite
> ordering.
>
> The concealment also reverses sign between corpora. On ISCX the macro average
> flatters the model by 13 points; on USTC it understates by one.

## Slide 16, padding attribution

> Back to padding. Padding-aware input is routinely credited with a large gain,
> and the decomposition shows where that gain lives.
>
> On ISCX at 1024 bytes the representation term, scalar byte to learned
> embedding, is worth plus 0.2408. The PAD-symbol term, telling the model which
> bytes are padding, is worth plus 0.0086. A factor of 28, and the PAD-symbol
> term is well inside the seed spread of 0.0373, so we report it as
> inconclusive rather than as a small positive effect. On USTC both terms are
> inside seed noise.
>
> The practical reading is short: use a learned byte embedding; the explicit
> padding symbol buys nothing we can measure. Zain will take the early-exit
> results.

---

# Part 4. Zain: the classifier, the split confound and the comparison

## Slide 17, accuracy per byte read

> This is the result the causal encoder was built for.
>
> Blue points are the fixed-depth heads, orange is the adaptive policy, and
> critically both come from the same trained weights. The only difference
> between a blue point and an orange point is whether the exit rule runs.
>
> At a threshold of 0.99 the policy reads 144 bytes per flow and scores 0.9659
> macro-F1, plus or minus 0.0113. The best fixed head, at 512 bytes, scores
> 0.9716 plus or minus 0.0124. The gap of 0.0058 is well inside the seed spread
> of either. The same accuracy, to within seed noise, for 28 percent of the
> byte cost. On ISCX the policy reads 116 bytes, 23 percent of 512, for 0.5134
> against 0.5474.
>
> Two things make that honest. It is against the model's own heads, so no
> architectural advantage is smuggled in. And the encoder is causal and tested,
> so head k cannot have seen byte k plus one. These are bytes genuinely never
> read, not a bookkeeping claim.

## Slide 18, the split confound

> This result was not in our original plan, and it is the one we would most
> want other people to act on.
>
> ISCX, six classes, the only ones the corpus can split group-wise, eight
> seeds. Classes, budget, model, seeds and split sizes all held fixed. The only
> thing that changes is where the train-test boundary falls. Moving from a
> group-wise split to a random within-capture split is worth plus 0.3782
> macro-F1.
>
> Put that next to the other effects in this talk. The entire model family was
> worth 0.0050. The padding symbol was worth 0.0086. Where the split falls is
> worth 0.3782, about seventy-five times the architecture effect.
>
> The mechanism is simple: a random split lets sessions from the same capture
> file land in both training and test, so the model memorises capture artefacts
> instead of the traffic class. None of the twelve reviewed studies states its
> split policy.

## Slide 19, comparison with the base papers

> Which brings us to the comparison, and to being honest about where we stand.
>
> Starred figures are quoted from the original publications under their own
> protocols; the last two rows are ours. Our ISCX figure, 0.5134, is far below
> the published ones, and we are not going to pretend otherwise.
>
> Three measured reasons. Class count: we report 11 ISCX classes where MH-Net
> and the ensemble study report six. Split policy, which we just showed is
> worth up to plus 0.3782 and which none of them states. And identifiers: the
> ET-BERT and YaTC rows come from an audit in one of our own base papers, and
> show ET-BERT falling from 0.960 to 0.510 and YaTC from 0.900 to 0.620 once
> identifier bytes are removed. Our gate removes them before the first run.
>
> So we do not claim to beat these numbers, and the table is laid out to make
> that clear. What we do claim is the budget: comparable work reads 784 to 1024
> bytes where we read 144.

---

# Part 5. Findings, limitations and conclusion

## Slide 20, key findings (Talha)

> Five things we can now state with evidence.
>
> One, the field reads five to seven times more of each flow than the
> measurements support: saturation at 512 bytes on USTC and 256 on ISCX against
> reviewed budgets of 784 to 1024 bytes.
>
> Two, past saturation reading more is measurably worse, not merely unhelpful:
> 0.9731 down to 0.9367 between 512 and 1500 bytes, McNemar at p below ten to
> the minus four.
>
> Three, saturation is a property of the traffic, not the model: a byte
> transformer and a 2017 CNN are not separable.
>
> Four, the padding gain is the byte embedding, not the padding symbol: plus
> 0.2408 against plus 0.0086.
>
> Five, an adaptive reader needs 144 bytes, not 512, with a tested causality
> guarantee behind the number.
>
> And the one that should change practice first: the train-test split alone is
> worth plus 0.3782 macro-F1, and none of the twelve studies states its policy.

## Slide 21, limitations (Zain)

> We want to be explicit about what this does not establish.
>
> The most serious threat is our own receptive field. It is 505 bytes, close
> enough to the 512-byte saturation point that the two could be confounded on
> the early-exit model. What excludes that is the transformer control, which
> has no such limit and saturates at the same packet budget anyway.
>
> Gap three is only partly answered: a lower bound on how much of each corpus
> is genuinely encrypted, not a measurement. Both corpora are from 2016, so a
> 512-byte saturation point may be a property of that cipher mix.
>
> On failures: ISCX accuracy is low in absolute terms, 0.4013 uncapped, and we
> report it rather than choosing a kinder protocol. `vpn_chat` at 0.042 recall
> is a genuine failure. The split study uses six classes, so that plus 0.3782
> carries a class-selection bias we state alongside it. And USTC cannot host
> the split study at all, because most of its classes come from one capture
> file.

## Slide 22, conclusion and future work (Talha)

> To conclude. This project took one number the literature copies from paper to
> paper and made it a measured variable.
>
> Macro-F1 saturates at 512 bytes on USTC and 256 on ISCX. Every budget used by
> twelve recent studies lies at or past that point, and on USTC reading further
> is significantly worse. A causal early-exit classifier comes within seed
> noise of the best fixed head while reading 28 and 23 percent of the bytes.
> And a random within-capture split is worth plus 0.3782 macro-F1, exceeding
> every architectural effect we measured.
>
> Future work follows from the limitations rather than from ambition. A corpus
> with more group-splittable classes would remove the class-selection bias. A
> larger length-balanced subset would turn the directional padding finding into
> a magnitude. Pre-training the byte transformer would test whether saturation
> moves once the representation is learned rather than fitted. And repeating
> the sweep on contemporary traffic would tell us whether 512 bytes is a
> property of these 2016 captures or of encrypted traffic generally.

## Slide 23, closing (Zain)

> Everything here is public: the full pipeline, the result JSON behind every
> number we quoted, the regression tests, and the sources of the paper, the
> poster and this deck. Both corpora are public too.
>
> Thank you for watching. I am Zain Shahid, i232582, and that was Muhammad
> Talha Arshad, i232548. We are happy to take questions.

---

## Delivery notes

- **Do not read the slides.** The slide carries the number; the narration
  carries the reason the number matters. Where the script repeats a figure
  that is on screen, it is because that figure is the point of the sentence.
- **Say the numbers carefully.** "Zero point nine seven three one", not "point
  nine seven". Precision is most of what this project argues for.
- **Name the handoffs.** Every transition between speakers is written into the
  script, at the end of slides 5, 11, 16 and 19, and in the headings of 20, 21,
  22 and 23. That is what makes the two-member requirement audible rather than
  something the marker has to infer.
- **Record in parts.** Each member's slide ranges are contiguous, so each can
  record their own segments separately and the parts can be joined in the
  editor. Cut on a slide advance, never mid-sentence.
- **One retake rule.** If a number comes out wrong, stop and redo that slide. A
  misspoken figure is the one error a marker will certainly catch.
