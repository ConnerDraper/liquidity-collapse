"""Beamer look for every figure: Latin Modern Sans, navy frame-title banner, 16:9 canvas.

Copy into a project and build every figure through frame() ... save().
Latin Modern Sans has no Greek letters: spell them out in figure text.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

FONT_DIR = Path.home() / ".local/share/fonts/latin-modern-sans"
FONT_FAMILY = "Latin Modern Sans"

BANNER = "#1b3a6b"
ACCENT = "#2e6db4"
INK = "#1b1b1b"
MUTED_INK = "#5a5a5a"
GRIDLINE = "#d9d9d9"

# Fixed colors, so a series keeps its color on every slide. Replace the keys with the
# project's own series; keep grey for benchmarks and baselines.
COLOR = {
    "dev": "#2a78d6", "holdout": "#eb6834", "after_scores": "#2a78d6", "raw": "#9a9a9a",
    "liquidity": "#1baf7a", "fund": "#5a5a5a", "baseline": "#5a5a5a", "benchmark": "#9a9a9a",
}

SLIDE = (10, 5.625)  # 16:9
BANNER_HEIGHT = 0.16
TITLE_WIDTH = 0.96
AXES_RECT = (0.09, 0.14, 0.62, 1 - BANNER_HEIGHT - 0.17)  # room for a legend on the right
WIDE_RECT = (0.09, 0.14, 0.86, 1 - BANNER_HEIGHT - 0.17)  # no side legend


def apply() -> None:
    fonts = sorted(FONT_DIR.glob("lmsans10-*.otf"))
    if not fonts:
        raise SystemExit(f"Latin Modern Sans is not installed in {FONT_DIR}; see the hpc-beamer-slides skill for the install loop.")
    for font in fonts:
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "axes.unicode_minus": True, "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY, "mathtext.sf": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic", "mathtext.bf": f"{FONT_FAMILY}:bold",
    })


def frame(title: str, wide: bool = False):
    """A slide: white canvas, banner with the frame title, one styled axes."""
    fig = plt.figure(figsize=SLIDE, facecolor="white")
    fig.patches.append(Rectangle((0, 1 - BANNER_HEIGHT), 1, BANNER_HEIGHT, transform=fig.transFigure,
                                 facecolor=BANNER, edgecolor="none", zorder=0))
    fig.patches.append(Rectangle((0, 1 - BANNER_HEIGHT - 0.012), 1, 0.012, transform=fig.transFigure,
                                 facecolor=ACCENT, edgecolor="none", zorder=0))
    banner_title = fig.text(0.04, 1 - BANNER_HEIGHT / 2, title, color="white", fontsize=19,
                            fontweight="bold", va="center", ha="left")
    width = banner_title.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    if width > TITLE_WIDTH - 0.04:
        banner_title.set_fontsize(19 * (TITLE_WIDTH - 0.04) / width)
    ax = fig.add_axes(WIDE_RECT if wide else AXES_RECT)
    ax.set_facecolor("white")
    ax.tick_params(colors=MUTED_INK, labelsize=11)
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED_INK)
    return fig, ax


def labels(ax, *, xlabel: str | None = None, ylabel: str) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK, fontsize=13, fontweight="bold")


def legend(ax) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1), frameon=False, fontsize=11)


def footnote(fig, text: str) -> None:
    fig.text(AXES_RECT[0], 0.03, text, color=MUTED_INK, fontsize=9, ha="left")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
