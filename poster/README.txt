Phase 5: research poster and standee
====================================

Deliverables
------------
  Phase5_Poster_A3.pdf          A3 landscape, 420 x 297 mm, 1 page
  Phase5_Standee_2.5x5ft.pdf    2.5 ft x 5 ft, 762 x 1524 mm, 1 page, full size

Both are built at final print size with no scaling required. Send either
straight to print; add bleed at the printer if the shop asks for it, since the
artwork runs edge to edge.

Source
------
  shared.css      the single design system: colour tokens, typography and the
                  component vocabulary (bands, cards, KPI callout, stat row,
                  figure frames, tables). Both artifacts link this one file, so
                  they cannot drift apart.
  poster.html     A3 layout, three columns
  standee.html    standee layout, one column
  figures/*.svg   vector figures, see below

Rebuilding
----------
  google-chrome --headless --disable-gpu --no-pdf-header-footer \
      --print-to-pdf=Phase5_Poster_A3.pdf poster.html
  google-chrome --headless --disable-gpu --no-pdf-header-footer \
      --print-to-pdf=Phase5_Standee_2.5x5ft.pdf standee.html

Page size comes from the @page rule in each file, so the output is exact.

Visual identity
---------------
Navy #0e2a4d bands on a near-white ground, with the accent colours taken
directly from src/nettc/plots.py (blue #2a78d6, orange #eb6834, green #1baf7a)
so the figures and the artwork around them share one palette. Type is Noto
Sans throughout. The same masthead, section bands, cards and callouts appear
in both pieces at different scale.

Figures
-------
All figures are vector SVG, so both pieces stay sharp at any print size.

  pipeline.svg    the methodology diagram, compiled from the same TikZ source
                  as Figure 1 of the paper
  the rest        regenerated from the Phase 3 result JSON by
                  scripts/make_poster_figures.py, on a reduced canvas so that
                  every label, including the ones plots.py sizes explicitly,
                  stays legible once the figure is scaled up for display
  qr_contact.svg  QR encoding mailto:i232548@isb.nu.edu.pk

Every number shown in either piece comes from the Phase 3 result JSON and
matches the Phase 4 paper.
