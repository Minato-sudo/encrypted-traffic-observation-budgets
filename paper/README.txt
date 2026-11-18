Phase 4: research paper source
==============================

Template
--------
Springer Nature LaTeX template (sn-jnl), taken from
https://www.overleaf.com/latex/templates/springer-nature-latex-template/myxmhdsbzkyd

The document class line is the template's own active line, unchanged:

    \documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

The preamble carries the same packages the template ships with, in the same
order, minus "listings" (no code listings appear in the paper) and plus
"tikz", which draws the pipeline diagram in Figure 1. The template marks that
point in the preamble as the place for additional packages.

Files
-----
  main.tex                  the paper
  references.bib            19 entries, generated from src/nettc/references.py
                            so the paper, the Phase 3 report and the code
                            repository cite the same works
  figures/                  vector PDF figures, produced by the Phase 3 runs
  sn-jnl.cls                Springer Nature class file, unmodified
  sn-mathphys-num.bst       Springer Nature numeric bibliography style
  Phase4_Research_Paper.pdf the compiled paper

Building
--------
On Overleaf: upload main.tex, references.bib and figures/ into the Springer
Nature template project and compile with pdfLaTeX. The project already holds
sn-jnl.cls and the .bst, so those two files here are only needed for an
offline build.

Offline (this project was built with Tectonic, which uses XeTeX):

    tectonic -X compile main.tex

Structure
---------
Main article: Sections 1-9 plus Declarations and Acknowledgements. Appendix A
(a synthetic positive control) and Appendix B (supporting table and figures)
follow, then the reference list. The appendix is the optional section the
Phase 4 brief allows for material that does not fit the main text; every claim
in the main article stands without it.

Provenance of the numbers
-------------------------
Every unstarred number in the paper comes from the Phase 3 result JSON in
../results/. Values marked * in Tables 1, 6 and 7 are quoted from the original
publications under their own protocols. Figures and tables are generated from
that JSON rather than transcribed, and the per-class and headline figures were
checked against it programmatically before submission.
