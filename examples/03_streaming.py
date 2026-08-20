"""A live loop over the RawSource protocol.

Nothing here knows the data is simulated. Swapping SimulatedSource for a scanner
adapter is the only change needed to run this against hardware.

    python examples/03_streaming.py
"""
import time

import numpy as np

import usx

probe = usx.presets.linear_5mhz()
medium = usx.Medium()
acq = usx.presets.acquisition(probe, depth=0.04, medium=medium)

# A phantom with a moving vessel, so successive frames actually differ.
phantom = usx.phantoms.flow_phantom(probe, medium, vessel_depth=0.022,
                                    peak_velocity=0.25, angle_deg=20.0,
                                    z_min=0.005, z_max=0.038)

config = usx.SimulatorConfig()
config.noise_db = -80.0
source = usx.ingest.SimulatedSource(
    probe, medium, acq, phantom,
    transmits=usx.presets.plane_wave_sequence(usx.Simulator(probe, medium, acq), 9.0, 3),
    config=config, n_frames=4)

grid = usx.grid_for(probe, 0.006, 0.038, 128, 320)
pipeline = usx.ImagingPipeline(grid)

print(f"{'frame':>6s} {'demod':>8s} {'beamform':>10s} {'post':>7s} {'total':>8s} {'fps':>7s}")
budget = []
for frame_index, channel_data in enumerate(source):
    image = pipeline.process_display(channel_data)
    s = pipeline.stats
    budget.append(s.total_ms)
    print(f"{frame_index:>6d} {s.demodulate_ms:>7.1f}ms {s.beamform_ms:>9.1f}ms "
          f"{s.postprocess_ms:>6.1f}ms {s.total_ms:>7.1f}ms {s.fps:>7.1f}")

print(f"\nsteady-state: {np.mean(budget[1:]):.1f} ms per frame "
      f"({1000/np.mean(budget[1:]):.1f} fps)")
print("the pipeline's smoothed brightness reference keeps the display stable")
print("across frames instead of re-normalising each one independently")
