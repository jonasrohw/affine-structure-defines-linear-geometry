# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research workspace for Hinrich Schütze, containing several independent projects. There is no unified build system — each subdirectory is a self-contained project.

## Project Subdirectories

### `mac/` — Bibliography & LaTeX Tools
Scripts for managing academic publication lists.

- `extract_my_papers.py`: Parses `anthology.bib` (ACL Anthology) and generates a LaTeX file listing all papers by Hinrich Schütze. Run from `mac/`:
  ```
  python extract_my_papers.py
  pdflatex my_acl_papers.tex && bibtex my_acl_papers && pdflatex my_acl_papers.tex && pdflatex my_acl_papers.tex
  ```
- `download_dblp.py`: Downloads DBLP data.
- LaTeX source for `definition_benchmark.tex` and related papers; compile with `pdflatex` + `bibtex`.

### `venice/` — Workshop Figure Generation
matplotlib scripts producing vector-diagram figures illustrating word embedding relationships.

- `create2.py` / `create_figures.py`: Generate PNG panels (left/middle/right). Run from the repo root:
  ```
  python venice/create2.py
  ```
  Output PNGs are saved to `venice/`.

### `vienna/` — Linear Representation Hypothesis (planned)
A planned PyTorch experiment studying whether LM internal representations of subjects transform linearly to object representations. The implementation does not yet exist; `project_plan.md` describes the full design:
- **Data**: Synthetic 201-token vocabulary (`a0..a99`, `b0..b99`, separator `x`), sequences like `x a5 b5 x a20 b20 ...`
- **Stack**: PyTorch transformer (GPT-style, 1–4 layers), `torch>=2.0`, `numpy`, `matplotlib`, `scikit-learn`
- **Key metric**: Delta cosine similarity (`Δcos = cos(o, ô) − cos(o, s)`) per Hernandez et al. (2024)
- Planned modules: `data_generation.py`, `model.py`, `training.py`, `representations.py`, `affine.py`, `evaluation.py`

### `igprezo/` — Claude Code/Work Presentation
LaTeX Beamer presentation introducing Claude Code and Claude Work. Source: `presentation.tex`. Figures `a.mBlTy.png`, `b.mBlTy.png` are screenshots used in slides.

### `arr202601/` — ACL Rolling Review (Jan 2026)
Email templates and author/reviewer contact information for ARR paper batch (papers 936, 1573, 4311, 4701, 5191, 6216, 6816, 8105).

### `christian/` — Personal Email Archive
~350 personal email text files; no code.

## Common LaTeX Compilation
```
pdflatex <file>.tex
bibtex <file>
pdflatex <file>.tex
pdflatex <file>.tex
```
