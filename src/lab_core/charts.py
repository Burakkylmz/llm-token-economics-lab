"""Shared, lab-agnostic matplotlib chart styling and I/O helpers.

Every lab's chart generator should use these instead of configuring
matplotlib itself, so every PNG this project produces shares one dark theme,
one validated color palette, and one "never fabricate missing data" policy.

Palette and dark-surface values are the validated defaults from this
project's dataviz design reference (categorical hues, chart chrome, and the
sequential blue ramp) -- not invented here. Colors are assigned by role
(categorical identity for reasoning_effort/prompt_type, sequential magnitude
for heatmap cells), never cycled arbitrarily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: this project only ever renders to PNG files.

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# -- Chart chrome (dark mode only -- these PNGs are never viewed in light mode) --
DARK_SURFACE = "#1a1a19"
DARK_PAGE = "#0d0d0d"
PRIMARY_INK = "#f0efec"
SECONDARY_INK = "#c3c2b7"
MUTED_INK = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"

# -- Categorical palette, dark-mode steps, fixed slot order (never cycled) --
CATEGORICAL_DARK = [
    "#3987e5",  # slot 1: blue
    "#d95926",  # slot 2: orange
    "#199e70",  # slot 3: aqua
    "#c98500",  # slot 4: yellow
    "#d55181",  # slot 5: magenta
    "#008300",  # slot 6: green
    "#9085e9",  # slot 7: violet
    "#e66767",  # slot 8: red
]

# Fixed, documented slot assignment for this project's two recurring
# categorical dimensions -- always the same color for the same level,
# across every chart, so a reader never has to re-learn the legend.
REASONING_EFFORT_COLOR = {
    "low": CATEGORICAL_DARK[0],
    "medium": CATEGORICAL_DARK[1],
    "high": CATEGORICAL_DARK[2],
}
REASONING_EFFORT_ORDER = ("low", "medium", "high")

PROMPT_TYPE_COLOR = {
    "short_prompt": CATEGORICAL_DARK[0],
    "long_prompt": CATEGORICAL_DARK[3],
}

# Sequential blue ramp (step 100 -> 700), for heatmap / magnitude encoding.
_SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def sequential_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("sequential_blue_dark", _SEQUENTIAL_BLUE_STEPS)


def configure_dark_theme() -> None:
    """Apply this project's dark chart theme to matplotlib's rcParams.

    Idempotent and cheap -- safe to call before building every figure.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_SURFACE,
            "axes.facecolor": DARK_SURFACE,
            "savefig.facecolor": DARK_SURFACE,
            "text.color": PRIMARY_INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": SECONDARY_INK,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "grid.color": GRIDLINE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.linewidth": 0.6,
            "axes.titlecolor": PRIMARY_INK,
            "legend.facecolor": DARK_SURFACE,
            "legend.edgecolor": BASELINE,
            "legend.labelcolor": SECONDARY_INK,
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )


def new_figure(figsize: tuple[float, float] = (8.0, 5.0), ncols: int = 1, nrows: int = 1, **subplot_kwargs):
    configure_dark_theme()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **subplot_kwargs)
    return fig, axes


def save_png(fig: Figure, path: Path, dpi: int = 200) -> Path:
    """Write `fig` as a high-resolution PNG at `path`, creating parent dirs.

    Never touches any other file -- this project's JSON/CSV/Markdown result
    files are read-only inputs to chart generation, never outputs of it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def no_data_text(ax, message: str = "Veri yok") -> None:
    """Mark an empty axes/cell honestly -- never render a missing measurement as zero."""
    ax.text(
        0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
        color=MUTED_INK, fontsize=9, style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def sample_size_caption(counts: Sequence[int]) -> Optional[str]:
    """Turkish caption warning against reading variability into single-repetition
    data, or None if every group has >= 2 observations.

    `counts` is the successful-call count per bar/condition being compared.
    """
    if not counts:
        return None
    if min(counts) < 2:
        return (
            "Not: bazı koşullarda tek ölçüm (n=1) bulunuyor -- bu değerler tanımlayıcıdır, "
            "istatistiksel anlamlılık iddiası taşımaz."
        )
    return None


def add_footer_caption(fig: Figure, text: Optional[str]) -> None:
    if not text:
        return
    fig.text(0.5, -0.02, text, ha="center", va="top", color=MUTED_INK, fontsize=8, wrap=True)
