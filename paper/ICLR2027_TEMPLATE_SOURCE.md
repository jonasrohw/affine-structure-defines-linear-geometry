# ICLR 2027 template provenance

The files `iclr2027_conference.sty`, `iclr2027_conference.bst`,
`iclr2027_conference.tex`, `iclr2027_conference.bib`, `fancyhdr.sty`,
`natbib.sty`, and `math_commands.tex` were copied from the official ICLR
2027 author kit:

https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

The archive was downloaded on 2026-09-01. Its SHA-256 digest is:

```text
0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e
```

The conference author guidelines are at:

https://iclr.cc/Conferences/2027/AuthorGuidelines

## Build the paper

From the repository root, regenerate the paper-specific summary figure and build
the manuscript with:

```bash
uv run --project experiments python paper/make_paper_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The submitted source is `main.tex`; `iclr2027_conference.tex` is the unmodified
conference example included for reference.
