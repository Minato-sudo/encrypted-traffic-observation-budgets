Phase 6: recorded presentation
==============================

Deliverables
------------
  Phase6_Presentation.pdf   25 pages, 16:9 (13.333 x 7.5 in), for screen:
                            23 slides for the talk, plus two backup slides
                            (B1, B2) held for questions about ISCX
  SCRIPT.md                 the voice-over script, about 15 and a half minutes,
                            split evenly between both members, with the handoffs
                            written in as spoken lines
  DEMO_RUNSHEET.md          the live terminal walkthrough for slide 11, with
                            the real commands and the output they actually
                            print
  RECORDING_GUIDE.md        OBS settings, how to join the segments, the upload
                            steps and the pre-upload checklist
  TALHA_PREP.md             Talha's own prep notes for slides 1-11: what each
                            slide means in plain terms, everyday analogies,
                            Roman Urdu where a concept is hard, the numbers not
                            to get wrong, and the questions each slide invites.
                            Not for recording, for understanding.

The video itself is not in the repository. Record it against SCRIPT.md, upload
it unlisted to YouTube, and submit the link.

Source
------
  slides.html               the deck. Links ../poster/shared.css for the
                            component vocabulary and the accent colours, so the
                            deck, the poster and the standee stay one family.
                            The deck overrides the dark chrome only, navy to
                            dark grey (#363b40), in its own :root block;
                            shared.css is untouched, so the printed poster and
                            standee keep their navy. To bring all three to grey,
                            change --navy and --navy-deep in shared.css instead
                            and delete the override here.
  figures/*.svg             vector figures, regenerated from the Phase 3
                            result JSON by scripts/make_slide_figures.py on a
                            reduced canvas so that the labels stay legible
                            once a figure is scaled to fill a slide.

Rebuilding
----------
  PYTHONPATH=src python3 scripts/make_slide_figures.py

  google-chrome --headless --disable-gpu --no-pdf-header-footer \
      --print-to-pdf=Phase6_Presentation.pdf slides.html

Page size comes from the @page rule in slides.html, so the output is exactly
16:9 and fills a 1920x1080 capture with no letterboxing.

What each member presents
-------------------------
One contiguous half each, with a single handoff at the end of slide 11.

  Talha (i232548)   slides 1-11                       about 7:28
                    the problem, the padding confound, the research gap, the
                    objectives, the whole methodology, the implementation, and
                    the live demonstration at the terminal

  Zain  (i232582)   slides 12-23                      about 8:04
                    every result, the comparison with the base papers, the key
                    findings, the limitations and the conclusion

The boundary falls where the talk turns: Talha finishes by showing the
measurement running, Zain takes everything it produced.

Because there is only one handoff, slide 1 and slide 23 each carry both voices,
so a marker who samples the video rather than watching it through still hears
that two people presented. Do not cut those lines.

Every number
------------
Every figure quoted in the deck and in the script comes from the Phase 3 result
JSON under results/ and matches the Phase 4 and Phase 7 paper. The padding
fractions on slide 3 are the mean padding fraction E[rho_k] of Eq. (2) in the
paper, not the fraction of sessions that contain any padding, which is a
different and larger quantity.
