#!/usr/bin/env python
"""Generate the GitHub social-preview banner (1280x640) -> docs/social-preview.png.

A clean, self-explanatory card built around the repo's hook: the inflated (wrong)
vs honest (right) evaluation contrast. Deterministic, no downloads.

Run: ``python scripts/make_social_preview.py``  (or ``make social``).
Then set it on GitHub: Settings → General → Social preview → upload docs/social-preview.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WRONG = "#d1495b"
RIGHT = "#2e8b57"
INK = "#1b2530"
MUTED = "#5b6b7a"


def main() -> int:
    # 1280x640 at 100 dpi = 12.8 x 6.4 inches.
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")

    # Left accent stripe.
    ax.add_patch(FancyBboxPatch((0, 0), 16, 640, boxstyle="square,pad=0",
                                facecolor=RIGHT, edgecolor="none"))

    # Title + pitch (left column).
    ax.text(70, 545, "ML & Signal Processing", fontsize=44, fontweight="bold", color=INK)
    ax.text(70, 487, "on Neural Signals 101", fontsize=44, fontweight="bold", color=INK)
    ax.text(72, 437, "raw EEG → features → models → an HONEST score",
            fontsize=18, color=MUTED)
    ax.text(72, 405, "A runnable, bilingual (EN / zh-TW) tutorial for beginners. CPU-only.",
            fontsize=15, color=MUTED)

    # Feature bullets.
    bullets = [
        "16 notebooks + 14 deep-dives, all execute on a laptop",
        "Leakage-safe evaluation is the default — no fake scores",
        "Teaches how NOT to fool yourself (Ch. 12 + 13)",
    ]
    for i, b in enumerate(bullets):
        y = 350 - i * 40
        ax.text(78, y, "▸", fontsize=17, color=RIGHT, fontweight="bold")
        ax.text(104, y, b, fontsize=16.5, color=INK)

    ax.text(72, 150, "github.com/ChiShengChen/neural-signals-101",
            fontsize=15, color=MUTED, fontweight="bold")
    ax.text(72, 116, "MIT · Python 3.11 · Open in Colab / nbviewer",
            fontsize=12.5, color=MUTED)

    # Right column: the inflated-vs-honest two-bar hook.
    bx = fig.add_axes([0.72, 0.18, 0.24, 0.62])
    bx.set_facecolor("#f5f7f9")
    for s in bx.spines.values():
        s.set_visible(False)
    bars = bx.bar([0, 1], [0.74, 0.66], width=0.62, color=[WRONG, RIGHT])
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["WRONG\n(leaky)", "RIGHT\n(honest)"], fontsize=12, color=INK)
    bx.set_ylim(0, 1.0)
    bx.set_yticks([])
    bx.axhline(0.5, ls="--", lw=1, color="gray")
    bx.text(0.5, 0.455, "chance", ha="center", fontsize=9, color="gray")
    for b, h in zip(bars, [0.74, 0.66]):
        bx.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", fontsize=13, fontweight="bold", color=INK)
    bx.set_title("Same model & data —\nonly the evaluation differs",
                 fontsize=11.5, color=INK)

    out = ROOT / "docs" / "social-preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    print(f"✅ saved {out} (1280x640)")
    print("   GitHub: Settings → General → Social preview → upload this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
