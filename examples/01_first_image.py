"""Simulate a phantom and beamform it — the shortest complete path through the engine.

    python examples/01_first_image.py
"""
import numpy as np

import usx

# 1. Describe the hardware and the tissue.
probe = usx.presets.linear_5mhz()
medium = usx.Medium()                       # 1540 m/s, 0.5 dB/(cm*MHz)
acq = usx.presets.acquisition(probe, depth=0.05, medium=medium)
print(f"probe:  {probe.n_elements} elements, {probe.center_frequency/1e6:.0f} MHz, "
      f"lambda = {probe.wavelength(medium.speed_of_sound)*1e6:.0f} um")
print(f"acq:    {acq.sampling_frequency/1e6:.0f} MHz, {acq.n_samples} samples, "
      f"PRF {acq.prf:.0f} Hz (unambiguous to {acq.max_depth(1540)*1e3:.0f} mm)")

# 2. Build a phantom: speckle background with three inclusions.
sim = usx.Simulator(probe, medium, acq)
phantom = usx.phantoms.cyst_phantom(probe, medium)
print(f"phantom: {len(phantom)} scatterers")

# 3. Transmit. Eleven steered plane waves, coherently compounded, give
#    focused-quality images at ultrafast frame rates.
transmits = sim.make_plane_waves(usx.angles_deg(-9, 9, 11))
raw = sim.simulate(transmits, phantom)
print(f"raw RF:  {raw.rf.shape} {raw.rf.dtype}  "
      f"({raw.rf.nbytes/1e6:.1f} MB, a zero-copy view of the engine's buffer)")

# 4. Reconstruct. The pipeline demodulates to IQ, beamforms, and reports timing.
grid = usx.grid_for(probe, z_min=0.005, z_max=0.05, n_lateral=256, n_axial=640)
pipeline = usx.ImagingPipeline(grid)
frame = pipeline.process_frame(raw)
print(f"frame:   {frame.data.shape} {frame.data.dtype}   <- complex, phase intact")
print(f"timing:  {pipeline.stats}")

# 5. Only now discard phase, for display.
bmode = usx.to_bmode(frame, dynamic_range_db=60, medium=medium)
usx.render.save_png("out/example_01.png", np.asarray(bmode).T, grid,
                    title="11-angle plane-wave compounding, DAS")
print("wrote out/example_01.png")

# 6. And measure it, rather than eyeballing it.
env = np.asarray(usx.envelope(frame))
stats = usx.metrics.summarize(env, grid,
                              lesion_xz=(0.0, 0.020), lesion_radius=0.003,
                              speckle_xz=(0.0, 0.045))
for k, v in stats.items():
    print(f"  {k:14s} {v: .3f}")
print(f"  (fully developed speckle has SNR {usx.metrics.RAYLEIGH_SPECKLE_SNR:.2f})")
