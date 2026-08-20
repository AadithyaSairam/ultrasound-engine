"""What the complex frame is for: velocity and strain.

Both estimators below read the *phase* of the beamformed IQ. Neither can be run
on a log-compressed greyscale image, which is why the engine's output type is a
complex frame.

    python examples/04_quantitative.py
"""
import numpy as np

import usx

probe = usx.presets.linear_5mhz()
medium = usx.Medium()

# ---------------------------------------------------------------- colour flow
acq = usx.presets.acquisition(probe, depth=0.04, medium=medium, prf=6000.0)
config = usx.SimulatorConfig()
config.noise_db = -80.0          # flow sits far closer to the noise floor than B-mode
sim = usx.Simulator(probe, medium, acq, config)

angle_deg, v_peak = 25.0, 0.30
phantom = usx.phantoms.flow_phantom(probe, medium, vessel_depth=0.022,
                                    vessel_radius=0.0035, angle_deg=angle_deg,
                                    peak_velocity=v_peak, z_min=0.006, z_max=0.038)

ensemble = usx.Simulator.repeat(sim.make_plane_waves([0.0]), 16)
iq = usx.demodulate(sim.simulate(ensemble, phantom), usx.DemodConfig())

grid = usx.grid_for(probe, 0.008, 0.038, 140, 340)
frames = usx.Beamformer().beamform_sequence(iq, grid, 1)

doppler = usx.DopplerConfig()
doppler.wall_filter = usx.WallFilter.POLYNOMIAL
doppler.kernel_axial = 6
flow = usx.color_flow(frames, acq.prf, medium.speed_of_sound, doppler)

valid = np.asarray(flow.valid).astype(bool)
v = np.asarray(flow.velocity)[valid]
expected = v_peak * np.sin(np.deg2rad(angle_deg))
print("colour flow")
print(f"  99th pct |v_axial|   {np.percentile(np.abs(v), 99)*100:6.1f} cm/s")
print(f"  expected             {abs(expected)*100:6.1f} cm/s"
      f"  (= {v_peak*100:.0f} x sin {angle_deg:.0f} deg)")
print(f"  Nyquist limit        {flow.nyquist_velocity*100:6.1f} cm/s")
print("  Doppler sees only the beam-aligned component — flow parallel to the")
print("  probe face produces no signal at all, which is why probes are angled.")

usx.render.save_png("out/example_04_flow.png",
                    usx.render.color_overlay(usx.render.bmode_image(frames[0], 50), flow),
                    grid, title="Colour flow")

# ---------------------------------------------------------------- elastography
acq2 = usx.presets.acquisition(probe, depth=0.055, medium=medium)
cfg2 = usx.SimulatorConfig()
cfg2.enable_motion = False
sim2 = usx.Simulator(probe, medium, acq2, cfg2)

inclusion, ratio, strain_applied = (0.0, 0.030, 0.005), 4.0, 0.01
before = usx.phantoms.elastography_phantom(probe, medium, inclusion, ratio, 0.0)
after = usx.phantoms.elastography_phantom(probe, medium, inclusion, ratio, -strain_applied)

txs = sim2.make_plane_waves(usx.angles_deg(-9, 9, 5))
grid2 = usx.grid_for(probe, 0.012, 0.050, 140, 420)
bf = usx.Beamformer()
fa = bf.beamform(usx.demodulate(sim2.simulate(txs, before), usx.DemodConfig()), grid2)
fb = bf.beamform(usx.demodulate(sim2.simulate(txs, after), usx.DemodConfig()), grid2)

disp = usx.estimate_axial_displacement(fa, fb, medium.speed_of_sound, 10, 3, 12)
strain = np.asarray(usx.axial_strain(disp, 25))
strain = np.where(np.asarray(disp.correlation) > 0.7, strain, np.nan)

X, Z = np.meshgrid(np.asarray(grid2.axis0), np.asarray(grid2.axis1), indexing="ij")
r2 = (X - inclusion[0]) ** 2 + (Z - inclusion[1]) ** 2
inside = r2 < (inclusion[2] * 0.6) ** 2
outside = (r2 > (inclusion[2] * 1.8) ** 2) & (np.abs(Z - inclusion[1]) < 0.010)

print("\nstrain elastography")
print(f"  |strain| in inclusion  {np.nanmean(np.abs(strain[inside]))*100:6.3f} %")
print(f"  |strain| in background {np.nanmean(np.abs(strain[outside]))*100:6.3f} %")
print(f"  measured ratio         {np.nanmean(np.abs(strain[outside]))/np.nanmean(np.abs(strain[inside])):6.2f}"
      f"  (applied {ratio:g})")
print("  Stiff tissue strains less under the same stress, so the strain map is a")
print("  stiffness map — palpation, quantified.")

lim = np.nanpercentile(np.abs(strain), 98)
usx.render.save_png("out/example_04_strain.png", strain.T * 100, grid2, cmap="viridis",
                    vmin=-lim * 100, vmax=0, colorbar_label="axial strain (%)",
                    title="Strain elastogram")
print("\nwrote out/example_04_flow.png and out/example_04_strain.png")
