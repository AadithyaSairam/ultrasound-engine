"""usx — a from-scratch ultrasound imaging engine.

The heavy lifting (simulation, demodulation, beamforming, Doppler) lives in a
C++ core exposed through :mod:`usx._usx`. This package re-exports that surface
and adds the parts that are more natural in Python: data ingest, image quality
metrics, rendering and a command line.

The engine's product is complex IQ, not a picture::

    import usx

    probe = usx.presets.linear_5mhz()
    sim   = usx.Simulator(probe, usx.Medium(), usx.presets.acquisition(probe))
    phan  = usx.phantoms.cyst_phantom(probe)

    raw   = sim.simulate(sim.make_plane_waves(usx.angles_deg(-9, 9, 11)), phan)
    grid  = usx.ScanGrid.cartesian(-0.019, 0.019, 256, 0.005, 0.055, 768)
    frame = usx.ImagingPipeline(grid).process_frame(raw)   # complex

    bmode = usx.to_bmode(frame)          # only now is phase discarded
"""

from __future__ import annotations

import numpy as np

from ._usx import (  # noqa: F401
    Acquisition,
    Beamformer,
    BeamformerConfig,
    ChannelData,
    ColorFlowMap,
    Combiner,
    Compounding,
    DemodConfig,
    DisplacementMap,
    DopplerConfig,
    Frame,
    GridType,
    ImagingPipeline,
    Medium,
    Phantom,
    PipelineConfig,
    PipelineStats,
    Probe,
    ScanConverted,
    ScanGrid,
    Scatterer,
    SimulatorConfig,
    Simulator,
    Transmit,
    TransmitType,
    Vec3,
    WallFilter,
    Window,
    apply_tgc,
    apply_wall_filter,
    axial_strain,
    color_flow,
    compute_transmit_delays,
    demodulate,
    design_lowpass,
    element_directivity,
    envelope,
    estimate_axial_displacement,
    log_compress,
    power_doppler,
    scan_convert,
    transmit_arrival_time,
    transmit_sensitivity,
    __version__,
)

from . import ingest, metrics, phantoms, presets, render  # noqa: E402,F401

__all__ = [n for n in dir() if not n.startswith("_")] + [
    "angles_deg", "to_bmode", "grid_for", "extent_mm",
]


def angles_deg(start: float, stop: float, count: int) -> list[float]:
    """Evenly spaced transmit angles, given in degrees, returned in radians.

    Steering angles are quoted in degrees everywhere in the literature and
    consumed in radians everywhere in the code; this is the one place that
    conversion should live.
    """
    if count == 1:
        return [np.deg2rad(0.5 * (start + stop))]
    return list(np.deg2rad(np.linspace(start, stop, count)))


def to_bmode(frame: Frame, dynamic_range_db: float = 60.0, gain_db: float = 0.0,
             medium: Medium | None = None, tgc: bool = True) -> np.ndarray:
    """Complex frame to a log-compressed image in [0, 1], shaped (axis0, axis1).

    This is the lossy step. Everything a quantitative estimator needs is gone
    afterwards, so call it last and keep the frame around.
    """
    env = envelope(frame)
    if tgc:
        med = medium or Medium()
        f0 = frame.demodulation_frequency or 5.0e6
        env = apply_tgc(env, frame.grid, med.attenuation, f0)
    return log_compress(env, dynamic_range_db, gain_db, -1.0)


def grid_for(probe: Probe, z_min: float = 0.005, z_max: float = 0.055,
             n_lateral: int = 256, n_axial: int = 768,
             width: float | None = None) -> ScanGrid:
    """A Cartesian grid spanning the probe's aperture over a depth range.

    Defaults to the aperture width, because reconstructing pixels the array
    never illuminated only adds clutter to the edges of the image.
    """
    half = 0.5 * (width if width is not None else probe.aperture_width())
    return ScanGrid.cartesian(-half, half, n_lateral, z_min, z_max, n_axial)


def extent_mm(grid: ScanGrid) -> tuple[float, float, float, float]:
    """matplotlib `extent` in millimetres for an image shaped (axis0, axis1).

    Images are stored lateral-major but displayed depth-down, so the array is
    transposed for display and the extent is ordered accordingly.
    """
    a0 = np.asarray(grid.axis0) * 1e3
    a1 = np.asarray(grid.axis1) * 1e3
    return (float(a0[0]), float(a0[-1]), float(a1[-1]), float(a1[0]))
