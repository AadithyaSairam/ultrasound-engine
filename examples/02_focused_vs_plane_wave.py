"""The frame-rate/quality trade, measured.

Classic focused imaging fires one transmit per scanline; plane-wave imaging fires
one transmit for the whole image. This script images the same phantom both ways
and reports what the difference actually costs, in transmits and in resolution.

    python examples/02_focused_vs_plane_wave.py
"""
import numpy as np

import usx

probe = usx.presets.linear_5mhz()
medium = usx.Medium()
acq = usx.presets.acquisition(probe, depth=0.045, medium=medium)
sim = usx.Simulator(probe, medium, acq)

phantom = usx.phantoms.wire_targets(depths=(0.015, 0.025, 0.035), lateral=(0.0,))
grid = usx.grid_for(probe, 0.010, 0.042, 200, 480)
bf_config = usx.BeamformerConfig()
bf_config.f_number = 1.75

panels, rows = [], []
for label, transmits, compounding in [
    ("focused, 64 lines", sim.make_focused_scan(64, -0.015, 0.015, 0.025),
     usx.Compounding.SELECT),
    ("1 plane wave", sim.make_plane_waves([0.0]), usx.Compounding.COHERENT),
    ("11 plane waves", sim.make_plane_waves(usx.angles_deg(-9, 9, 11)),
     usx.Compounding.COHERENT),
]:
    raw = sim.simulate(transmits, phantom)
    iq = usx.demodulate(raw, usx.DemodConfig())

    cfg = usx.BeamformerConfig()
    cfg.f_number = 1.75
    cfg.compounding = compounding
    frame = usx.Beamformer(cfg).beamform(iq, grid)

    env = np.asarray(usx.envelope(frame))
    psf = usx.metrics.measure_psf(env, grid, (0.0, 0.025))

    # Frame rate is set by how many round trips the sequence needs.
    max_fps = medium.speed_of_sound / (2 * 0.045 * len(transmits))
    rows.append((label, len(transmits), max_fps, psf.lateral_fwhm * 1e3,
                 psf.axial_fwhm * 1e3))
    panels.append((f"{label}\n{max_fps:.0f} fps max", np.asarray(
        usx.log_compress(env, 50.0)).T))

usx.render.save_comparison("out/example_02.png", panels, grid,
                           suptitle="Transmit scheme: frame rate against resolution")
print("wrote out/example_02.png\n")
print(f"{'scheme':>20s} {'transmits':>10s} {'max fps':>9s} "
      f"{'lateral mm':>11s} {'axial mm':>9s}")
for label, n, fps, lat, ax in rows:
    print(f"{label:>20s} {n:>10d} {fps:>9.0f} {lat:>11.3f} {ax:>9.3f}")
print("\nOne plane wave is 64x faster than the focused scan and clearly worse.")
print("Eleven plane waves are still ~6x faster and recover most of the resolution:")
print("compounding synthesises a transmit focus at every depth, where the focused")
print("scan only has one focal zone.")
