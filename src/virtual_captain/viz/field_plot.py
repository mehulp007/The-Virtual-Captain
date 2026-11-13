import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from virtual_captain.field.geometry import BOUNDARY_RADIUS, PITCH_HALF, ring_outline
from virtual_captain.viz import style


def draw_field(ax: plt.Axes, radius: float = BOUNDARY_RADIUS, labels: bool = True) -> plt.Axes:
    """Empty ground, viewed from behind the striker with the bowler's end at the top."""
    ax.add_patch(Circle((0, 0), radius, fill=False, lw=1.2, ec=style.AXIS))
    rx, ry = ring_outline()
    ax.plot(rx, ry, lw=0.8, ls=(0, (4, 3)), color=style.AXIS)
    ax.add_patch(
        Rectangle((-1.5, -PITCH_HALF), 3.0, 2 * PITCH_HALF, fc=style.GRID, ec="none", zorder=0)
    )
    if labels:
        kw = {"fontsize": 7, "color": style.INK_MUTED, "ha": "center", "va": "center"}
        ax.text(radius * 0.78, -radius * 0.98, "off", **kw)
        ax.text(-radius * 0.78, -radius * 0.98, "leg", **kw)
    pad = radius * 1.08
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
