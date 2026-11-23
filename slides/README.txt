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
  Talha (i232548)   slides 1-5, 11-15, 20, 22        about 8:09
                    problem, padding confound, research gap, objectives; the
                    live demonstration at the terminal; the leakage gate,
                    budget response, architecture control and per-class
                    behaviour; key findings; conclusion

  Zain  (i232582)   slides 6-10, 16-19, 21, 23        about 7:21
                    methodology and implementation; padding attribution,
                    accuracy per byte, the split confound and the comparison
                    with the base papers; limitations; closing

Talha drives the terminal, so slide 11 opens his second block rather than
closing Zain's. The split also keeps each member's results beside the material
they introduced: Talha frames the four gaps and reports the results answering
them, Zain builds the padding modes and the early-exit encoder and reports what
those measured.

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
