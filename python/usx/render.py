"""Rendering B-mode, colour flow and strain images.

Colour choices here follow clinical convention rather than a house style,
because the conventions carry meaning to anyone who reads ultrasound: greyscale
for B-mode with depth increasing downwards; red for flow toward the transducer
and blue for flow away (the "BART" convention — Blue Away, Red Toward); a warm
map for power Doppler, which has no direction to encode.

Note that the engine's internal sign convention is the opposite of the display
one: positive axial velocity means motion *away* from the probe. The flip
happens here, once, and nowhere else.
"""

from __future__ import annotations

import numpy as np

from ._usx import Frame, ScanGrid, envelope, log_compress


def _extent(grid: ScanGrid):
    a0 = np.asarray(grid.axis0) * 1e3
    a1 = np.asarray(grid.axis1) * 1e3
    return (float(a0[0]), float(a0[-1]), float(a1[-1]), float(a1[0]))


def bmode_image(frame: Frame, dynamic_range_db: float = 60.0, gain_db: float = 0.0) -> np.ndarray:
    """Log-compressed B-mode in [0, 1], oriented for display (depth down)."""
    img = log_compress(envelope(frame), dynamic_range_db, gain_db, -1.0)
    return np.asarray(img).T


def color_overlay(bmode: np.ndarray, flow, velocity_scale: float | None = None,
                  power_floor_db: float = -22.0, percentile: float = 99.0) -> np.ndarray:
    """Composite a colour flow map over a B-mode image, returning RGB in [0, 1].

    Three rules make a colour box readable, and two of them are about *not*
    painting:

    - pixels the estimator marked invalid keep their greyscale value rather than
      being given a colour;
    - the colour is blended in proportion to power, so vessel edges fade instead
      of ending in a hard, fake-looking boundary;
    - the velocity scale defaults to a high percentile of what was actually
      measured, not to the Nyquist limit. Scaling to Nyquist is what makes real
      colour boxes look washed out: if the flow only reaches a third of the
      limit, two thirds of the colour bar goes unused. Clinical scanners expose
      this as the "scale" or "velocity range" control, and setting it is most of
      the skill in getting a usable colour image.
    """
    v = np.asarray(flow.velocity).T
    p = np.asarray(flow.power).T
    valid = np.asarray(flow.valid).T.astype(bool)

    if velocity_scale is None:
        vv = np.abs(v[valid])
        velocity_scale = float(np.percentile(vv, percentile)) if vv.size else 1.0
        velocity_scale = max(velocity_scale, 1e-6)

    # Display convention: toward the probe (negative axial velocity) is warm.
    u = np.clip(-v / velocity_scale, -1.0, 1.0)

    pmax = p[valid].max() if valid.any() else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        p_db = 10.0 * np.log10(np.where(p > 0, p / (pmax + 1e-30), 1e-12))
    alpha = np.clip((p_db - power_floor_db) / (-power_floor_db), 0.0, 1.0)
    alpha = np.where(valid, alpha, 0.0)

    grey = np.clip(bmode, 0.0, 1.0)
    rgb = np.dstack([grey, grey, grey])

    # The standard clinical ramp: dark red to yellow toward the probe, dark blue
    # to cyan away from it. Both legs run from dark to bright with increasing
    # speed, so magnitude reads even in a greyscale reproduction.
    t = np.abs(u)
    toward = u > 0
    bright = 0.45 + 0.55 * t                     # dark at low speed, bright at high
    mid = np.clip(1.6 * (t - 0.35), 0.0, 1.0)    # green channel, shared by both legs
    flow_rgb = np.dstack([
        np.where(toward, bright, 0.0),           # red   -> toward the probe
        mid,                                     # green -> yellow / cyan at speed
        np.where(toward, 0.0, bright),           # blue  -> away from the probe
    ])

    a = alpha[..., None]
    return np.clip(rgb * (1 - a) + flow_rgb * a, 0.0, 1.0)


def save_png(path, image, grid: ScanGrid | None = None, title: str | None = None,
             cmap: str = "gray", vmin=None, vmax=None, colorbar_label: str | None = None,
             dpi: int = 150, figsize=(5.0, 6.0)):
    """Write an image to a PNG with millimetre axes and an equal aspect ratio.

    Equal aspect is not cosmetic: an ultrasound image with unequal lateral and
    axial scales silently misrepresents the shape of everything in it, and
    lesion shape is diagnostic.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asarray(image)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    kw = dict(aspect="equal", origin="upper", interpolation="bilinear")
    if grid is not None:
        kw["extent"] = _extent(grid)
    if img.ndim == 2:
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
        if colorbar_label:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(colorbar_label)
    else:
        ax.imshow(img, **kw)
    ax.set_xlabel("lateral (mm)")
    ax.set_ylabel("depth (mm)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def save_comparison(path, panels, grid: ScanGrid | None = None, suptitle: str | None = None,
                    dpi: int = 150, panel_size=(3.2, 4.4)):
    """A row of labelled panels sharing one depth axis.

    Beamformer comparisons are meaningless side by side unless the panels share
    a dynamic range and a scale, so this enforces both.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]), dpi=dpi,
                             squeeze=False)
    for ax, (label, img) in zip(axes[0], panels):
        arr = np.asarray(img)
        kw = dict(aspect="equal", origin="upper", interpolation="bilinear")
        if grid is not None:
            kw["extent"] = _extent(grid)
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=1, **kw)
        else:
            ax.imshow(arr, **kw)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("lateral (mm)")
    axes[0][0].set_ylabel("depth (mm)")
    for ax in axes[0][1:]:
        ax.set_yticklabels([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path
