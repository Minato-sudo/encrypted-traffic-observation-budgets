Phase 6: recorded presentation
==============================

Deliverables
------------
  Phase6_Presentation.pdf   16 pages, 16:9 (13.333 x 7.5 in), for screen:
                            14 slides for the talk, plus two backup slides
                            (B1, B2) held for questions about ISCX
  SCRIPT.md                 the voice-over script, about 14 minutes, one
                            handoff, with Talha's half written short and spoken
                            and both members' lines on slides 1 and 14
  DEMO_RUNSHEET.md          the live terminal walkthrough for slide 11, with
                            the real commands and the output they actually
                            print
  RECORDING_GUIDE.md        OBS settings, how to join the segments, the upload
                            steps and the pre-upload checklist
  Talha_Script_Slides_1-6.pdf
                            Talha's half as a printable A4 reading script,
                            4 pages: large type, numbers highlighted, the demo
                            as a run-this / say-this table, and the handoff
                            cue marked. Source is TALHA_SCRIPT_PRINT.html.
  TALHA_PREP.md             Talha's own prep notes for slides 1-6: what each
                            slide means in plain terms, everyday analogies,
                            Roman Urdu where a concept is hard, the numbers not
                            to get wrong, and the questions each slide invites.
                            Not for recording, for understanding.

The video itself is not in the repository. Record it against SCRIPT.md, upload
it unlisted to YouTube, and submit the link.

Source
------
  slides.html               the deck, self-contained: its CSS is in the file.
                            The layout follows the proposal deck supplied as a
                            design reference (accent eyebrow, large headline,
                            white cards on a tinted page, large coloured
                            figures, a takeaway line, a small source line; no
                            header band and no footer bar). Dark chrome is dark
                            grey (#33383d), not navy. The accent trio is
                            unchanged from src/nettc/plots.py, so the artwork
                            still matches the colours inside the figures it
                            frames. This deck no longer shares poster/shared.css
                            with the Phase 5 poster and standee, which keep
                            their own design.
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
One contiguous half each, with a single handoff at the end of slide 6.

  Talha (i232548)   slides 1-6                        about 6:13
                    the problem, the padding confound, the research gap, the
                    objectives, the whole methodology, the implementation, and
                    the live demonstration at the terminal

  Zain  (i232582)   slides 7-14                       about 8:04
                    every result, the comparison with the base papers, the key
                    findings, the limitations and the conclusion

The boundary falls where the talk turns: Talha finishes by showing the
measurement running, Zain takes everything it produced.

Because there is only one handoff, slide 1 and slide 14 each carry both voices,
so a marker who samples the video rather than watching it through still hears
that two people presented. Do not cut those lines.

Every number
------------
Every figure quoted in the deck and in the script comes from the Phase 3 result
JSON under results/ and matches the Phase 4 and Phase 7 paper. The padding
fractions on slide 3 are the mean padding fraction E[rho_k] of Eq. (2) in the
paper, not the fraction of sessions that contain any padding, which is a
different and larger quantity.
