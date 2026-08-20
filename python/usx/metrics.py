"""Image quality metrics.

Every number here is a way of asking "is this beamformer actually better, or
does it just look sharper?". That distinction matters because the adaptive
methods in this engine are non-linear: they can improve every metric on this
list while making the image less useful, by suppressing the speckle texture that
a reader relies on to judge tissue. So the metrics are reported together, and
gCNR is included specifically because it is the one that cannot be gamed by a
monotone intensity transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _mask_circle(grid, center_xz, radius, inner_scale=1.0):
    x = np.asarray(grid.axis0)[:, None]
    z = np.asarray(grid.axis1)[None, :]
    r = np.hypot(x - center_xz[0], z - center_xz[1])
    return r <= radius * inner_scale, r


def contrast_db(env, grid, center_xz, radius, background=(1.6, 2.6)) -> float:
    """Mean-power ratio between a lesion and an annulus of background, in dB.

    Simple, standard, and easy to inflate: any transform that darkens low
    amplitudes improves it without improving detectability.
    """
    inside, r = _mask_circle(grid, center_xz, radius, 0.7)
    outside = (r > radius * background[0]) & (r < radius * background[1])
    env = np.asarray(env, dtype=np.float64)
    pin = np.mean(env[inside] ** 2)
    pout = np.mean(env[outside] ** 2)
    return float(10.0 * np.log10((pin + 1e-30) / (pout + 1e-30)))


def cnr(env, grid, center_xz, radius, background=(1.6, 2.6)) -> float:
    """Contrast-to-noise ratio: |mu_in - mu_out| / sqrt(var_in + var_out).

    Accounts for speckle variance, which plain contrast ignores. Still not
    invariant to non-linear intensity mappings.
    """
    inside, r = _mask_circle(grid, center_xz, radius, 0.7)
    outside = (r > radius * background[0]) & (r < radius * background[1])
    env = np.asarray(env, dtype=np.float64)
    a, b = env[inside], env[outside]
    denom = np.sqrt(a.var() + b.var())
    return float(abs(a.mean() - b.mean()) / (denom + 1e-30))


def gcnr(env, grid, center_xz, radius, background=(1.6, 2.6), bins: int = 100) -> float:
    """Generalized CNR: 1 - the overlap of the two regions' distributions.

    This is the metric to trust when comparing non-linear beamformers. It is the
    probability that a randomly chosen pixel can be correctly assigned to lesion
    or background, so it is bounded in [0, 1] and — crucially — invariant under
    any monotone transform of the pixel values. A beamformer that "improves"
    contrast by stretching the dynamic range moves CNR and leaves gCNR alone.

    The invariance is only exact if the binning is too. Histogramming the raw
    values on equal-width bins is the common implementation and it is *not*
    invariant: a power-law transform crowds samples into different bins and
    shifts the answer by several percent. Binning on the pooled *ranks* instead
    makes the estimator exactly invariant, since a monotone map leaves every
    rank unchanged.

    Rodriguez-Molares et al., "The generalized contrast-to-noise ratio",
    IEEE IUS 2018.
    """
    inside, r = _mask_circle(grid, center_xz, radius, 0.7)
    outside = (r > radius * background[0]) & (r < radius * background[1])
    env = np.asarray(env, dtype=np.float64)
    a, b = env[inside], env[outside]
    if a.size == 0 or b.size == 0:
        return 0.0

    pooled = np.sort(np.concatenate([a, b]))
    # Rank of each sample within the pooled set, mapped to [0, 1].
    ra = np.searchsorted(pooled, a, side="left") / max(pooled.size - 1, 1)
    rb = np.searchsorted(pooled, b, side="left") / max(pooled.size - 1, 1)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ha, _ = np.histogram(ra, bins=edges)
    hb, _ = np.histogram(rb, bins=edges)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(1.0 - np.minimum(ha, hb).sum())


def speckle_snr(env, grid, region_xz, half_size, detrend: bool = True) -> float:
    """mu / sigma of the envelope in a uniform region.

    Fully developed speckle is Rayleigh-distributed, for which this ratio is
    exactly 1.91. A measured value near 1.91 confirms the phantom really is
    fully developed (enough scatterers per resolution cell); a value well above
    it means the beamformer has smoothed the speckle away, which is a warning
    that resolution has been traded for a cleaner-looking texture.

    `detrend` removes the mean brightness variation along each axis first, and
    it matters more than it sounds. Attenuation alone puts a 10 dB gradient
    across a 10 mm deep region, and the transmit aperture tapers off at the
    lateral edges; both inflate the measured standard deviation and drag the
    ratio well below 1.91 even when the speckle itself is perfectly Rayleigh.
    Removing them measures the speckle instead of the system's gain profile.
    """
    x = np.asarray(grid.axis0)
    z = np.asarray(grid.axis1)
    mx = np.abs(x - region_xz[0]) <= half_size[0]
    mz = np.abs(z - region_xz[1]) <= half_size[1]
    v = np.asarray(env, dtype=np.float64)[np.ix_(mx, mz)]
    if v.size == 0:
        return float("nan")
    if detrend and v.shape[0] > 1 and v.shape[1] > 1:
        v = v / np.maximum(v.mean(axis=0, keepdims=True), 1e-30)
        v = v / np.maximum(v.mean(axis=1, keepdims=True), 1e-30)
    return float(v.mean() / (v.std() + 1e-30))


RAYLEIGH_SPECKLE_SNR = 1.9137


@dataclass
class PointSpreadFunction:
    x: float
    z: float
    lateral_fwhm: float
    axial_fwhm: float
    peak: float
    sidelobe_db: float

    def __repr__(self) -> str:
        return (f"PSF(x={self.x*1e3:.2f}mm z={self.z*1e3:.2f}mm "
                f"lateral={self.lateral_fwhm*1e3:.3f}mm axial={self.axial_fwhm*1e3:.3f}mm "
                f"sidelobe={self.sidelobe_db:.1f}dB)")


def _fwhm(axis, profile) -> float:
    """Full width at half maximum with linear interpolation of the crossings."""
    profile = np.asarray(profile, dtype=np.float64)
    i = int(np.argmax(profile))
    half = profile[i] * 0.5
    if profile[i] <= 0:
        return float("nan")

    def crossing(direction):
        j = i
        while 0 <= j + direction < len(profile) and profile[j + direction] > half:
            j += direction
        k = j + direction
        if not (0 <= k < len(profile)):
            return float(axis[j])
        t = (half - profile[j]) / (profile[k] - profile[j])
        return float(axis[j] + t * (axis[k] - axis[j]))

    return abs(crossing(1) - crossing(-1))


def measure_psf(env, grid, near_xz, search_radius: float = 0.002) -> PointSpreadFunction:
    """Measure the point spread function of the target nearest `near_xz`.

    Reported alongside the contrast metrics, because contrast and resolution are
    the two ends of the same trade and quoting one without the other says
    nothing.
    """
    env = np.asarray(env, dtype=np.float64)
    ax0 = np.asarray(grid.axis0)
    ax1 = np.asarray(grid.axis1)
    _, r = _mask_circle(grid, near_xz, search_radius)
    win = r <= search_radius
    if not win.any():
        raise ValueError("no pixels within the search radius")
    masked = np.where(win, env, -np.inf)
    i, j = np.unravel_index(np.argmax(masked), masked.shape)

    lat = _fwhm(ax0, env[:, j])
    axi = _fwhm(ax1, env[i, :])

    # Highest value outside two beam widths of the peak: the sidelobe floor that
    # limits how well a weak target next to a strong one can be seen.
    dist = np.hypot(ax0[:, None] - ax0[i], ax1[None, :] - ax1[j])
    far = dist > max(3 * lat, 3 * axi, 5e-4)
    peak = env[i, j]
    # If the search window is not large enough to contain anything outside the
    # mainlobe there is no sidelobe measurement to make. Reporting a huge
    # negative dB figure instead would look like an excellent result.
    region = far & win
    side_db = (float(20 * np.log10((env[region].max() + 1e-30) / (peak + 1e-30)))
               if region.any() else float("nan"))
    return PointSpreadFunction(
        x=float(ax0[i]), z=float(ax1[j]), lateral_fwhm=lat, axial_fwhm=axi,
        peak=float(peak), sidelobe_db=side_db)


def summarize(env, grid, lesion_xz=None, lesion_radius=None, wire_xz=None,
              speckle_xz=None, speckle_half=(0.004, 0.004)) -> dict:
    """Everything at once, as a plain dict, for comparing beamformer settings."""
    out = {}
    if lesion_xz is not None and lesion_radius:
        out["contrast_db"] = contrast_db(env, grid, lesion_xz, lesion_radius)
        out["cnr"] = cnr(env, grid, lesion_xz, lesion_radius)
        out["gcnr"] = gcnr(env, grid, lesion_xz, lesion_radius)
    if wire_xz is not None:
        psf = measure_psf(env, grid, wire_xz)
        out["lateral_fwhm_mm"] = psf.lateral_fwhm * 1e3
        out["axial_fwhm_mm"] = psf.axial_fwhm * 1e3
        out["sidelobe_db"] = psf.sidelobe_db
    if speckle_xz is not None:
        out["speckle_snr"] = speckle_snr(env, grid, speckle_xz, speckle_half)
    return out
