#!/usr/bin/env python3
"""Render a generated Markdown report as one self-contained, print-ready page.

There is no LaTeX or pandoc on this machine, so the submission PDF is produced
by opening this HTML and printing to PDF (Ctrl+P, "Save as PDF", A4). The page
embeds every figure as a data URI, so the single file carries the whole
document and can be moved or attached without losing its images.

Needs the pure-Python `markdown` package:

    pip install markdown
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import sys
from pathlib import Path

CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.5; color: #15171a;
  max-width: 46em; margin: 0 auto; padding: 2em 1.5em;
  background: #fff;
}
h1 { font-size: 20pt; line-height: 1.25; margin: 0 0 .2em; letter-spacing: -.01em; }
h2 {
  font-size: 14pt; margin: 2.2em 0 .6em; padding-bottom: .25em;
  border-bottom: 1px solid #d8dbdf; page-break-after: avoid;
}
h3 { font-size: 11.5pt; margin: 1.6em 0 .4em; page-break-after: avoid; }
h4 { font-size: 10.5pt; margin: 1.2em 0 .3em; color: #33383d; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
code, pre { font-family: "DejaVu Sans Mono", "Menlo", monospace; font-size: 9pt; }
code { background: #f3f4f6; padding: .1em .3em; border-radius: 3px; }
pre {
  background: #f7f8fa; border: 1px solid #e3e6ea; border-radius: 4px;
  padding: .8em 1em; overflow-x: auto; page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 1em 0;
  font-size: 9.2pt; page-break-inside: avoid;
}
th, td { border: 1px solid #d8dbdf; padding: .38em .6em; text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: .8em auto; }
blockquote { margin: 1em 0; padding-left: 1em; border-left: 3px solid #d8dbdf; color: #4a4f55; }
hr { border: none; border-top: 1px solid #d8dbdf; margin: 2em 0; }
em { color: #33383d; }
details { margin: .8em 0; }
summary { cursor: pointer; font-size: 9.5pt; color: #4a4f55; }

/* Each numbered component starts a fresh page, so the marker can find them. */
h2 { page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
@media screen { h2 { page-break-before: auto; } }
"""

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def embed_images(html: str, base: Path) -> tuple[str, int, int]:
    """Replace every local <img src> with a base64 data URI."""
    embedded = missed = 0

    def one(match: re.Match) -> str:
        nonlocal embedded, missed
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = Path(src)
        if not path.is_absolute():
            path = base / src
        if not path.exists():
            missed += 1
            return match.group(0)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        embedded += 1
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', one, html), embedded, missed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", required=True, help="the generated .md to render")
    ap.add_argument("--out", default=None, help="output .html (default: alongside)")
    ap.add_argument("--base", default=".",
                    help="directory that relative image paths resolve against")
    args = ap.parse_args()

    try:
        import markdown
    except ImportError:
        print("this needs the `markdown` package:  pip install markdown", file=sys.stderr)
        return 1

    src = Path(args.markdown)
    text = src.read_text()
    title = next((ln.lstrip("# ").strip() for ln in text.split("\n")
                  if ln.startswith("# ")), src.stem)

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "attr_list", "sane_lists", "md_in_html"],
    )
    body, embedded, missed = embed_images(body, Path(args.base))

    out = Path(args.out) if args.out else src.with_suffix(".html")
    out.write_text(HTML.format(title=title, css=CSS, body=body))
    size = out.stat().st_size / 1e6
    print(f"wrote {out} ({size:.1f} MB, {embedded} figures embedded"
          + (f", {missed} NOT FOUND" if missed else "") + ")")
    print("to get the PDF: open it and print to PDF (A4, background graphics on)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
