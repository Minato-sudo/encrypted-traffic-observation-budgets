Phase 6: recorded presentation
==============================

Deliverables
------------
  Phase6_Presentation.pdf   25 pages, 16:9 (13.333 x 7.5 in), for screen:
                            23 slides for the talk, plus two backup slides
                            (B1, B2) held for questions about ISCX
  SCRIPT.md                 the full voice-over script, split between both
                            members, with handoffs and delivery notes
  DEMO_RUNSHEET.md          the live terminal walkthrough for slide 11, with
                            the real commands and the output they actually
                            print
  RECORDING_GUIDE.md        OBS settings, how to join the segments, the upload
                            steps and the pre-upload checklist

The video itself is not in the repository. Record it against SCRIPT.md, upload
it unlisted to YouTube, and submit the link.

Source
------
  slides.html               the deck. Links ../poster/shared.css, which is the
                            same design system the Phase 5 poster and standee
                            use, so the three pieces cannot drift apart.
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
  Talha (i232548)   slides 1-5, 12-16, 20, 22
                    problem, padding confound, research gap, objectives; the
                    leakage gate, budget response, architecture control,
                    per-class behaviour and padding attribution; key findings;
                    conclusion

  Zain  (i232582)   slides 6-11, 17-19, 21, 23
                    methodology, implementation and the live demonstration;
                    accuracy per byte, the split confound and the comparison
                    with the base papers; limitations; closing

Backup slides
-------------
Pages 24 and 25 are numbered B1 and B2 and are not part of the 23-slide talk.
They carry the ISCX budget response and the ISCX accuracy-per-byte frontier,
which the narration quotes but does not show, so they are there for the
question most likely to be asked: "and on the other corpus?". Stop advancing at
slide 23 during the recording.

Every number
------------
Every figure quoted in the deck and in the script comes from the Phase 3 result
JSON under results/ and matches the Phase 4 and Phase 7 paper. The padding
fractions on slide 3 are the mean padding fraction E[rho_k] of Eq. (2) in the
paper, not the fraction of sessions that contain any padding, which is a
different and larger quantity.
