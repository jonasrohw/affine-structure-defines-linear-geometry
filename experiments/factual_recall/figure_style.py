"""Shared ACL figure typography, semantic colors, and publication exports.

Dimensions match the unmodified ACL template (16 cm text, 0.6 cm gutter).
Keep figures at these widths in LaTeX so their 8 pt labels are not rescaled.
Captions are typeset separately by ACL at 10 pt.
"""

from pathlib import Path

import matplotlib as mpl

TEXT_WIDTH = 16 / 2.54
COLUMN_WIDTH = 7.7 / 2.54
FONT_SIZE = 8
EXPORT_DPI = 1200
COLORS = {
    "structured": "#69BDB0",  # sea glass
    "random": "#F098A0",      # coral rose
    "lavender": "#AF9BDC",
    "apricot": "#EDB477",
    "blue": "#88B9DF",
    "ink": "#293445",
    "muted": "#647184",
    "grid": "#E5E9EF",
    "pale": "#F5F7FA",
}


def apply_style():
    """Use the same Times/Computer Modern pairing as main.tex, via LaTeX."""
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{times}",
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{times}",
        "font.family": "serif",
        # Let the preamble select Times text. Matplotlib's "Times" alias also
        # loads mathptmx, unlike ACL's times package, and changes math symbols.
        "font.serif": ["Computer Modern Roman"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.titlepad": 9,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "axes.edgecolor": COLORS["muted"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        # Keep tick strings portable across the TeX export backends.
        "axes.unicode_minus": False,
        "axes.prop_cycle": mpl.cycler(color=[COLORS[k] for k in
            ("structured", "random", "lavender", "apricot", "blue")]),
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 1,
        "lines.linewidth": 1.35,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.6,
        "hatch.linewidth": 0.45,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.35,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": EXPORT_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
    })


def save_figure(figure, stem):
    """Preserve physical size: vector PDF/SVG plus a 1200-DPI PNG."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # pdfLaTeX embeds the fonts directly; Matplotlib's DVI-to-PDF font mapping
    # can lose math glyphs (notably minus signs) with this pinned runtime.
    figure.savefig(stem.with_suffix(".pdf"), backend="pgf", dpi=EXPORT_DPI)
    for extension in ("svg", "png"):
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=EXPORT_DPI)
