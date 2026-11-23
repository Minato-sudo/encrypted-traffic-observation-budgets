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

**One handoff, halfway through.**

| Member | Slides | Covers | Time |
|---|---|---|---|
| **Talha** (i232548) | **1&ndash;11** | the problem, the padding confound, the research gap, the objectives, the whole methodology, the implementation, and the **live demonstration** | 7:28 |
| **Zain** (i232582) | **12&ndash;23** | every result, the comparison with the base papers, the key findings, the limitations and the conclusion | 8:04 |

The boundary falls where the talk itself turns. Talha sets up the problem,
explains how the measurement was built, and finishes by showing it running at
the terminal. Zain then takes everything the measurement produced. 36 seconds
apart, which is as even as a clean single cut gets on 23 slides.

**One safeguard, because there is only one handoff.** The brief will not accept a
video carried by one member, and with a single switch at the 7-minute mark a
marker who samples the first few minutes hears only one voice. So **Zain speaks
one line on slide 1** as well, introducing himself and saying which half is his,
and **both members speak on slide 23**. That puts both voices inside the first
minute and the last.

**This is the 15-minute version.** 2,130 spoken words at a normal narration pace
of 150 words per minute is 14 minutes 10 seconds, plus a live demonstration of
about 1 minute 20 seconds, paced by the commands rather than by the words: a
total of about **15 and a half minutes**. Talha speaks 7:28 including the
demonstration and Zain 8:04.

Nothing is cut from the project to get here. All 23 slides are presented, both
members present problem, method, results and conclusion, and all seven of the
required topics are covered. What is cut is the elaboration: each slide is
stated once and moved on from, and the slide itself carries the detail a viewer
may want to pause on.

Every number spoken below is on the slide it belongs to, and all of them come
from the result JSON in `results/`. Nothing is rounded differently between the
deck, the paper and this script.

The deck is 25 pages: **slides 1 to 23 are the talk**, and pages 24 and 25 are
backup slides B1 and B2, carrying the ISCX budget response and the ISCX frontier
for questions. Stop advancing at slide 23 while recording.

---

# Part 1. Talha: slides 1 to 11, the problem, the method and the demonstration

## Slide 1, title (0:26)

> **Talha:** Good afternoon. Our Computer Networks research project asks one
> question: how many bytes does encrypted traffic classification actually need?
> I am Muhammad Talha Arshad, i232548, and I will take the problem, the research
> gap, the methodology and a live demonstration of the code.
>
> **Zain:** And I am Zain Shahid, i232582. I will take the second half: every
> result, the comparison with the published work, and the conclusions.

## Slide 2, the problem (0:42)

> A deep traffic classifier does not see a flow. It sees the first B bytes of it,
> zero-padded to a fixed length. That prefix is the observation budget.
>
> It is not a knob on the side, it is the operating point: a 1024-byte budget
> cannot decide before 1024 bytes arrive, and a middlebox holds B bytes for every
> concurrent flow.
>
> Across twelve studies from 2025 and 2026 the budget runs from five to twenty
> packets, or 784 to 1024 bytes, and in not one is it measured. Zero of twelve
> report accuracy as a function of the budget. That curve is what this project is
> for.

## Slide 3, the padding confound (0:34)

> It is worse than an unexamined hyperparameter.
>
> Flows are short and heavy-tailed. The median ISCX session is 274 bytes, the
> median USTC session 474, so a larger budget reads more padding, not more
> traffic: at 512 bytes the average ISCX session is 48 percent padding.
>
> Any gain from a larger budget therefore mixes two effects, more real bytes and
> more padding to use as a length cue, and nobody separates them. A paper
> concluding 1024 bytes works best may have measured a length classifier.

## Slide 4, the research gap (0:27)

> Four gaps come out of the literature, and we address them in this order.
>
> One, the budget is inherited rather than measured. Two, aggregate accuracy
> conceals per-class failure. Three, corpora are called encrypted without anyone
> measuring how much of them is. Four, shortcut learning from addresses and ports
> is rarely excluded, and one of our base papers shows leaving those bytes in is
> worth up to 0.45 accuracy.

## Slide 5, objectives (0:41)

> Five objectives, each with a measurable answer.
>
> One, measure the budget response and locate the saturation budget: the
> smallest budget beyond which no larger one is reliably better. Two, test
> whether that is a property of the model, by repeating the sweep with a
> different architecture. Three, decompose the padding effect into a
> representation term and a padding-symbol term. Four, build a classifier that
> reads only what it needs, with a causality guarantee. Five, report honestly:
> exclude shortcut learning, report per-class behaviour, and quantify what the
> split alone explains.
>
> Two corpora, 385 thousand sessions, 31 classes. Here is how we built it.

## Slide 6, the pipeline (0:31)

> The design requirement follows from the objective: if the budget is a measured
> variable, it must be the only thing that changes between two points on the
> curve.
>
> So one pass writes a fixed 2048-byte HDF5 session store and every budget is a
> slice of it. Parsing, sessionization, deduplication and anonymisation happen
> once, before any budget exists, so nothing can drift. Only the dashed region
> repeats per budget. Splits are group-wise by capture file, every point three
> seeds.

## Slide 7, the leakage gate (0:35)

> Gap four is usually handled with a sentence promising identifiers were removed.
> We made it a precondition the pipeline enforces.
>
> The gate is the condition on screen: a probe trained only on the anonymised
> identifier bytes must score no higher than the majority-class rate plus a
> margin of 0.1. If it does, the anonymisation has failed and the pipeline halts.
> It does not warn, it stops.
>
> `id_positions` is the gate; `ident` measures what the shortcut would have been
> worth if left in; `ports` masks the port bytes.

## Slide 8, padding decomposed (0:34)

> For the padding question we built five input modes. `zero` is the baseline,
> each byte as a scalar. `embed` swaps in a learned embedding but leaves padding
> as zero. `mask` keeps the embedding and adds a distinct PAD symbol. `convmask`
> also makes the convolution ignore padded positions. And `length_balanced`
> removes length as a cue altogether.
>
> The literature changes representation and padding symbol together, then credits
> the whole gain to padding-aware input. Comparing zero to embed to mask
> separates them, each term checked against the seed spread.

## Slide 9, the causal early-exit classifier (1:00)

> This is where the project could most easily have fooled itself.
>
> The idea is simple: heads at 64, 128, 256, 512 and 1024 bytes; at inference,
> stop at the first head whose confidence clears a threshold tau, and charge that
> flow only the bytes it consumed. But the claim holds only if head k is
> incapable of having seen byte k plus one.
>
> So the encoder is six causal dilated convolution blocks, dilations one through
> 32, left-padding only, receptive field 505 bytes. And the normalisation is
> per-position, `ChannelNorm1d`, never BatchNorm or GroupNorm: both average over
> the length axis, which would let a byte at position 900 shift the activation at
> position 10.
>
> We prove it by induction over the layer index and we test it: 505 bytes of
> forward reach, zero backwards. The test exists because a one-line change there
> would destroy the property while the accuracy numbers stayed plausible.

## Slide 10, implementation (0:37)

> Fifteen modules, from a streaming pcap reader through sessionization, the
> store, the slicer, four model families and the experiment layer. 385 thousand
> sessions out of about 55 gigabytes of captures.
>
> Reproducibility here is mechanical, not aspirational. Every table and figure is
> generated from the result JSON by `make_report.py`, so no number is typed by
> hand in the paper, the poster or this deck. And five regression tests,
> including one that plants a cue at a known byte depth and requires the sweep to
> find it.
>
> Let me show you some of this running.

## Slide 11, live demonstration (about 1:20)

> **[Switch to the terminal. Follow `DEMO_RUNSHEET.md`. Let the commands set the
> pace; the lines below are what to say over them, not a paragraph to read.]**
>
> > First the receptive-field probe. One byte changes at position 100. Positions
> > 100 to 604 move, so forward reach is 505 bytes, the analytic figure, and
> > backward reach is zero.
>
> > Now the causality test. The second block is the one that matters: head k fed
> > only its own prefix gives logits identical to head k fed the whole sequence.
>
> > Here is the encoder, and the line the guarantee rests on, `ChannelNorm1d`.
>
> > The USTC budget sweep, straight out of its JSON: 0.9731 at 512 bytes, 0.9564
> > at 784, 0.9367 at 1500. The curve goes down.
>
> > The significance file for the same comparison: 2488 sessions correct at 512
> > and wrong at 784, against 94 the other way.
>
> > And the generated report. Every table here comes from those JSON files, which
> > is why nothing in the deck was typed by hand.
>
> > That is the implementation. Zain will take you through what it measured.
>
> **[Switch back to the slides and hand over to Zain.]**

---

# Part 2. Zain: slides 12 to 23, every result and the conclusions

## Slide 12, the leakage gate result (0:39)

> Thank you Talha. I am Zain Shahid, i232582, and I will take the results.
>
> Starting with the gate, because nothing else means anything without it.
>
> On USTC a probe trained on the anonymised identifier positions reaches 0.1351
> accuracy against a majority-class rate of 0.1359, so below chance; on ISCX,
> 0.1095 against 0.2258. The identifier bytes are provably unlearnable and the
> gate opens.
>
> The same probe trained on those fields *before* anonymisation reaches 0.53 on
> ISCX. More than half the ISCX label is recoverable from addresses and ports
> with no payload at all. That is Gap four, measured.

## Slide 13, the budget response (0:52)

> This is the central result. Macro-F1 against byte budget on USTC, three seeds,
> shaded band is the seed spread.
>
> Saturation falls at 512 bytes; on ISCX, 256. The vertical markers are the
> twelve reviewed studies' budgets, and every one sits at or past it.
>
> But look past that point. Macro-F1 goes 0.9731 at 512 bytes, 0.9564 at 784,
> 0.9367 at 1500. It falls. The obvious objection is seed noise, so we tested it:
> McNemar's exact test on the paired 512 against 784 comparison finds 2488
> sessions correct at 512 and wrong at 784, against 94 the other way, p below ten
> to the minus four.
>
> So reading more is not the safe choice people treat it as. Past saturation it
> is measurably worse, because the extra bytes are mostly padding.

## Slide 14, the architecture control (0:38)

> There is an obvious objection and we take it seriously: maybe a 2017
> convolutional network simply cannot use bytes past 512, and a modern
> architecture would.
>
> So we replaced the whole convolutional stack with a byte transformer,
> self-attention over byte embeddings, on the same store, split and seeds. The
> two families are not separable: 0.0050 macro-F1 apart at peak against a seed
> spread of 0.0379, and both saturate at two packets.
>
> Saturation is therefore a property of where the discriminative bytes lie, not
> of the network reading them. That is the finding that generalises.

## Slide 15, per-class behaviour (0:36)

> Gap two was that the aggregate conceals per-class failure. Here is the concrete
> form.
>
> ISCX at 784 bytes, uncapped: accuracy 0.4013, macro-F1 0.5110, weighted-F1
> 0.3798. `vpn_chat` is the second-largest class in the test set, 686 sessions,
> at 0.042 recall. Its mirror image is `vpn_file`, recall 1.000 at precision
> 0.241, acting as a sink for sessions the model cannot place.
>
> These are not thin classes: the 44-session `vpn_email` reaches 0.955 recall.
> And the concealment reverses sign between corpora, flattering the model by 13
> points on ISCX and understating by one on USTC.

## Slide 16, padding attribution (0:33)

> Back to padding, where the decomposition shows what the gain actually is.
>
> On ISCX at 1024 bytes the representation term, scalar byte to learned
> embedding, is worth plus 0.2408. The PAD-symbol term is worth plus 0.0086. A
> factor of 28, and the smaller term sits well inside the seed spread of 0.0373,
> so we report it as inconclusive. On USTC both terms are inside seed noise.
>
> The reading is short: use a learned byte embedding; the padding symbol buys
> nothing we can measure.

## Slide 17, accuracy per byte read (0:48)

> This is the result the causal encoder was built for.
>
> Blue is the fixed-depth heads, orange the adaptive policy, and both come from
> the same trained weights. The only difference is whether the exit rule runs.
>
> At a threshold of 0.99 the policy reads 144 bytes per flow for 0.9659 macro-F1,
> against 0.9716 for the 512-byte head. The gap of 0.0058 is well inside the seed
> spread of either: the same accuracy, to within seed noise, for 28 percent of the
> byte cost. On ISCX the policy reads 116 bytes, 23 percent of 512.
>
> And it is honest twice over: against the model's own heads, so no architectural
> advantage is smuggled in, and causal, so these are bytes genuinely never
> read.

## Slide 18, the split confound (0:43)

> This was not in our original plan, and it is the result we would most want
> other people to act on.
>
> ISCX, six classes, eight seeds, with classes, budget, model and split sizes all
> held fixed. The only thing that changes is where the train-test boundary falls,
> and going from a group-wise split to a random within-capture one is worth plus
> 0.3782 macro-F1.
>
> Put that next to the other effects here. The entire model family was worth
> 0.0050; the padding symbol, 0.0086. The split is worth 0.3782, seventy-five
> times the architecture effect, because a random split lets sessions from one
> capture file land in both training and test.

## Slide 19, comparison with the base papers (0:50)

> Which brings us to being honest about where we stand.
>
> Starred figures are quoted from the original publications under their own
> protocols; the last two rows are ours. Our ISCX figure, 0.5134, is far below
> the published ones, for three measured reasons. Class count: we report 11 ISCX classes where MH-Net
> reports six. Split policy, worth up to plus 0.3782, which none of them states.
> And identifiers: these ET-BERT and YaTC rows come from an audit in one of our
> own base papers, showing ET-BERT falling 0.960 to 0.510 and YaTC 0.900 to 0.620
> once identifier bytes are removed.
>
> So we do not claim to beat these numbers. What we claim is the budget:
> comparable work reads 784 to 1024 bytes where we read 144.

## Slide 20, key findings (0:44)

> Five things we can now state with evidence.
>
> One, the field reads five to seven times more of each flow than the
> measurements support.
>
> Two, past saturation reading more is measurably worse, not merely unhelpful:
> 0.9731 down to 0.9367, at p below ten to the minus four.
>
> Three, saturation is a property of the traffic, not the model.
>
> Four, the padding gain is the byte embedding, not the padding symbol: plus
> 0.2408 against plus 0.0086.
>
> Five, an adaptive reader needs 144 bytes, not 512.
>
> And the one that should change practice first: the train-test split alone is
> worth plus 0.3782, and none of the twelve studies states its policy.

## Slide 21, limitations (0:45)

> We want to be explicit about what this does not establish.
>
> The most serious threat is our own receptive field. At 505 bytes it is close
> enough to the 512-byte saturation point that the two could be confounded on the
> early-exit model. What excludes that is the transformer control, which has no
> such limit and saturates at the same packet budget anyway.
>
> Gap three is only partly answered, a lower bound rather than a measurement.
> Both corpora are from 2016. ISCX accuracy is low in absolute terms and we
> report it rather than choose a kinder protocol. And the split study uses six
> classes, so the plus 0.3782 carries a class-selection bias.

## Slide 22, conclusion and future work (0:37)

> To conclude. This project took one number the literature copies from paper to
> paper and made it a measured variable, on two corpora, with a classifier that
> reads 28 percent of the bytes for the same accuracy and a causality guarantee
> behind that claim.
>
> Future work follows from the limitations rather than from ambition: a corpus
> with more group-splittable classes, a larger length-balanced subset,
> pre-training the byte transformer, and repeating the sweep on contemporary
> traffic, to find out whether 512 bytes is a property of these 2016 captures or
> of encrypted traffic.

## Slide 23, closing (0:19)

> **Zain:** Everything here is public: the pipeline, the result JSON behind every
> number we quoted, the tests, and the sources of the paper, the poster and this
> deck. I am Zain Shahid, i232582.
>
> **Talha:** And Muhammad Talha Arshad, i232548. Thank you for watching, and we
> are happy to take questions.

---

## Delivery notes

- **Do not read the slides.** The slide carries the number; the narration
  carries the reason the number matters. This version is deliberately lean, so
  each sentence is load-bearing: dropping one loses a step in the argument.
- **Say the numbers carefully.** "Zero point nine seven three one", not "point
  nine seven". Precision is most of what this project argues for.
- **There is only one handoff, so make it unmissable.** It is at the end of
  slide 11: Talha says "Zain will take you through what it measured", and Zain
  opens slide 12 by naming himself and his roll number. Do not shorten either
  line. With a single switch it is the only point in the middle of the video
  where a marker can hear that two people presented, which is why slide 1 and
  slide 23 both carry two voices as well.
- **Record in parts.** Each member has one contiguous range, so each can record
  alone and the parts are joined in the editor. Slide 1 and slide 23 need both
  members at the microphone; everything between is one voice at a time. Cut on a
  slide advance, never mid-sentence.
- **One retake rule.** If a number comes out wrong, stop and redo that slide. A
  misspoken figure is the one error a marker will certainly catch.
- **If you want the longer version back**, the material that was trimmed is the
  elaboration, not the findings: slides 2, 9, 13, 15, 17 and 19 each had a
  further paragraph of reasoning, and slide 21 listed two more failure cases.
  Nothing in this version contradicts the paper, so any of it can be spoken from
  the paper directly.
