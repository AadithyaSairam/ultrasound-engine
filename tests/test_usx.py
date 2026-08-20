"""Python-level tests.

The C++ suite (core/tests/test_main.cpp) covers the physics. These tests cover
the boundary: that numpy views really are views, that the ingest protocol is
interchangeable, that metrics behave the way their definitions promise, and that
the whole thing survives a round trip through a file.
"""

from __future__ import annotations

import numpy as np
import pytest

import usx


@pytest.fixture(scope="module")
def setup():
    probe = usx.presets.linear_5mhz()
    probe.n_elements = 64
    medium = usx.Medium()
    acq = usx.presets.acquisition(probe, depth=0.04, medium=medium)
    cfg = usx.SimulatorConfig()
    cfg.noise_db = -200.0
    sim = usx.Simulator(probe, medium, acq, cfg)
    return probe, medium, acq, sim


def test_probe_geometry():
    p = usx.presets.linear_5mhz()
    x = np.asarray(p.element_x)
    assert len(x) == p.n_elements
    assert np.allclose(x.mean(), 0.0, atol=1e-9), "the array is centred on x = 0"
    assert np.allclose(np.diff(x), p.pitch, rtol=1e-5), "elements are evenly pitched"
    # Half-wavelength pitch is the grating-lobe-free limit for a steering array.
    assert usx.presets.phased_3mhz().pitch < 0.55 * usx.presets.phased_3mhz().wavelength(1540)


def test_acquisition_covers_the_requested_depth():
    p = usx.presets.linear_5mhz()
    for depth in (0.03, 0.06, 0.12):
        a = usx.presets.acquisition(p, depth=depth)
        record = a.n_samples / a.sampling_frequency
        assert record >= 2 * depth / 1540.0, "the record is long enough for the round trip"
        assert a.max_depth(1540.0) >= depth, "the PRF does not alias the requested depth"


def test_rf_view_is_zero_copy(setup):
    probe, medium, acq, sim = setup
    ph = usx.Phantom.point_targets([usx.Vec3(0, 0, 0.02)])
    cd = sim.simulate(sim.make_plane_waves([0.0]), ph)
    a = cd.rf
    assert a.shape == (1, probe.n_elements, acq.n_samples)
    assert a.dtype == np.float32
    # Writing through the view must reach the C++ buffer.
    a[0, 0, 0] = 12345.0
    assert cd.rf[0, 0, 0] == pytest.approx(12345.0)
    assert not a.flags.owndata, "the array views the engine's buffer rather than copying it"


def test_frame_is_complex_and_localizes(setup):
    probe, medium, acq, sim = setup
    target = usx.Vec3(0.004, 0.0, 0.025)
    cd = sim.simulate(sim.make_plane_waves([0.0]), usx.Phantom.point_targets([target]))
    grid = usx.ScanGrid.cartesian(0.0, 0.008, 81, 0.021, 0.029, 81)
    frame = usx.ImagingPipeline(grid).process_frame(cd)
    data = frame.data
    assert data.dtype == np.complex64, "the engine's output keeps phase"
    env = np.abs(data)
    i, j = np.unravel_index(env.argmax(), env.shape)
    assert np.asarray(grid.axis0)[i] == pytest.approx(target.x, abs=2e-4)
    assert np.asarray(grid.axis1)[j] == pytest.approx(target.z, abs=2e-4)


def test_bmode_is_normalized(setup):
    probe, medium, acq, sim = setup
    ph = usx.phantoms.wire_targets(depths=(0.02, 0.03), lateral=(0.0,))
    cd = sim.simulate(sim.make_plane_waves([0.0]), ph)
    grid = usx.grid_for(probe, 0.01, 0.04, 64, 160)
    frame = usx.ImagingPipeline(grid).process_frame(cd)
    img = np.asarray(usx.to_bmode(frame, dynamic_range_db=50))
    assert img.shape == (64, 160)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert img.max() > 0.9


def test_resolution_improves_with_more_angles(setup):
    probe, medium, acq, sim = setup
    ph = usx.Phantom.point_targets([usx.Vec3(0.0, 0.0, 0.025)])
    grid = usx.ScanGrid.cartesian(-0.004, 0.004, 161, 0.023, 0.027, 81)
    bf = usx.Beamformer(usx.BeamformerConfig())

    widths = {}
    for n in (1, 11):
        txs = usx.presets.plane_wave_sequence(sim, 12.0, n)
        iq = usx.demodulate(sim.simulate(txs, ph), usx.DemodConfig())
        env = np.asarray(usx.envelope(bf.beamform(iq, grid)))
        widths[n] = usx.metrics.measure_psf(env, grid, (0.0, 0.025)).lateral_fwhm

    # Compounding steered plane waves synthesises a transmit focus, so the
    # two-way beam narrows towards the receive-only limit. This is the entire
    # justification for the technique.
    assert widths[11] < widths[1] * 0.75, (
        f"compounding should sharpen the beam: {widths[1]*1e3:.3f} -> {widths[11]*1e3:.3f} mm")


def test_gcnr_is_invariant_to_monotone_transforms(setup):
    """The property that makes gCNR the metric to trust for non-linear beamformers."""
    probe, medium, acq, sim = setup
    grid = usx.ScanGrid.cartesian(-0.008, 0.008, 60, 0.018, 0.032, 120)
    rng = np.random.default_rng(0)
    x = np.asarray(grid.axis0)[:, None]
    z = np.asarray(grid.axis1)[None, :]
    inside = np.hypot(x - 0.0, z - 0.025) < 0.003
    env = np.abs(rng.normal(size=(60, 120)) + 1j * rng.normal(size=(60, 120)))
    env = np.where(inside, env * 0.15, env)

    g1 = usx.metrics.gcnr(env, grid, (0.0, 0.025), 0.003)
    g2 = usx.metrics.gcnr(np.sqrt(env), grid, (0.0, 0.025), 0.003)
    g3 = usx.metrics.gcnr(env ** 2.5, grid, (0.0, 0.025), 0.003)
    assert g1 == pytest.approx(g2, abs=0.05)
    assert g1 == pytest.approx(g3, abs=0.05)
    assert 0.2 < g1 <= 1.0

    # Plain CNR, by contrast, moves under exactly the same transforms.
    c1 = usx.metrics.cnr(env, grid, (0.0, 0.025), 0.003)
    c3 = usx.metrics.cnr(env ** 2.5, grid, (0.0, 0.025), 0.003)
    assert abs(c1 - c3) > 0.05 * c1


def test_speckle_snr_matches_rayleigh(setup):
    probe, medium, acq, sim = setup
    ph = usx.phantoms.cyst_phantom(probe, medium, cysts=(), z_min=0.010, z_max=0.035,
                                   density_per_cell=15.0)
    txs = usx.presets.plane_wave_sequence(sim, 9.0, 5)
    iq = usx.demodulate(sim.simulate(txs, ph), usx.DemodConfig())
    grid = usx.grid_for(probe, 0.015, 0.030, 96, 240)
    env = np.asarray(usx.envelope(usx.Beamformer().beamform(iq, grid)))
    snr = usx.metrics.speckle_snr(env, grid, (0.0, 0.0225), (0.005, 0.005))
    # Fully developed speckle is Rayleigh, for which mu/sigma = 1.91.
    assert snr == pytest.approx(usx.metrics.RAYLEIGH_SPECKLE_SNR, rel=0.15), (
        f"speckle SNR {snr:.2f} is far from Rayleigh — the phantom may not have "
        "enough scatterers per resolution cell")


def test_cyst_is_darker_than_background(setup):
    probe, medium, acq, sim = setup
    ph = usx.phantoms.cyst_phantom(probe, medium, cysts=((0.0, 0.025, 0.003, 0.0),),
                                   z_min=0.010, z_max=0.038)
    txs = usx.presets.plane_wave_sequence(sim, 9.0, 7)
    iq = usx.demodulate(sim.simulate(txs, ph), usx.DemodConfig())
    grid = usx.grid_for(probe, 0.015, 0.035, 128, 240)
    env = np.asarray(usx.envelope(usx.Beamformer().beamform(iq, grid)))
    stats = usx.metrics.summarize(env, grid, lesion_xz=(0.0, 0.025), lesion_radius=0.003)
    assert stats["contrast_db"] < -6.0
    assert stats["gcnr"] > 0.5


def test_ingest_sources_are_interchangeable(tmp_path, setup):
    probe, medium, acq, sim = setup
    ph = usx.phantoms.wire_targets(depths=(0.025,), lateral=(0.0,))
    txs = sim.make_plane_waves([0.0])

    src = usx.ingest.SimulatedSource(probe, medium, acq, ph, txs,
                                     config=sim_config_quiet(), n_frames=2)
    frames = list(src.frames())
    assert len(frames) == 2

    path = usx.ingest.save_npz(tmp_path / "acq.npz", frames[0])
    file_src = usx.ingest.FileSource(path)
    replayed = next(file_src.frames())

    # A round trip through the file must preserve both samples and geometry.
    assert np.array_equal(np.asarray(frames[0].rf), np.asarray(replayed.rf))
    assert replayed.probe.n_elements == probe.n_elements
    assert replayed.acq.sampling_frequency == pytest.approx(acq.sampling_frequency)
    assert len(replayed.transmits) == len(txs)
    assert np.allclose(replayed.transmits[0].delays, txs[0].delays, atol=1e-12)

    # And beamforming the replayed data must give the same image.
    grid = usx.ScanGrid.cartesian(-0.004, 0.004, 41, 0.022, 0.028, 61)
    bf = usx.Beamformer()
    a = np.asarray(usx.envelope(bf.beamform(usx.demodulate(frames[0], usx.DemodConfig()), grid)))
    b = np.asarray(usx.envelope(bf.beamform(usx.demodulate(replayed, usx.DemodConfig()), grid)))
    assert np.allclose(a, b, rtol=1e-5, atol=1e-6)


def sim_config_quiet():
    c = usx.SimulatorConfig()
    c.noise_db = -200.0
    return c


def test_array_source_adapter(setup):
    probe, medium, acq, sim = setup
    ph = usx.phantoms.wire_targets(depths=(0.025,), lateral=(0.0,))
    txs = sim.make_plane_waves([0.0])
    raw = np.asarray(sim.simulate(txs, ph).rf).copy()

    src = usx.ingest.ArraySource([raw, raw], probe, medium, acq, txs)
    got = list(src.frames())
    assert len(got) == 2
    assert np.array_equal(np.asarray(got[0].rf), raw)


def test_doppler_recovers_a_known_velocity(setup):
    probe, medium, acq, sim = setup
    v_true = 0.12
    ph = usx.Phantom.speckle_box(-0.004, 0.004, 0.022, 0.028, 1e-4, 3000, 5)
    ph.velocities = np.tile([0.0, 0.0, v_true], (len(ph), 1)).astype(np.float32)

    seq = usx.Simulator.repeat(sim.make_plane_waves([0.0]), 12)
    iq = usx.demodulate(sim.simulate(seq, ph), usx.DemodConfig())
    grid = usx.ScanGrid.cartesian(-0.003, 0.003, 21, 0.023, 0.027, 61)
    frames = usx.Beamformer().beamform_sequence(iq, grid, 1)

    cfg = usx.DopplerConfig()
    cfg.wall_filter = usx.WallFilter.NONE
    flow = usx.color_flow(frames, acq.prf, medium.speed_of_sound, cfg)
    valid = np.asarray(flow.valid).astype(bool)
    v = np.asarray(flow.velocity)[valid]
    assert valid.sum() > 50
    assert np.median(v) == pytest.approx(v_true, abs=0.02)
    assert flow.nyquist_velocity > v_true


def test_lateral_flow_produces_no_axial_doppler(setup):
    """Doppler measures only the beam-aligned component — the angle problem."""
    probe, medium, acq, sim = setup
    ph = usx.Phantom.speckle_box(-0.006, 0.006, 0.022, 0.028, 1e-4, 3000, 5)
    ph.velocities = np.tile([0.20, 0.0, 0.0], (len(ph), 1)).astype(np.float32)

    seq = usx.Simulator.repeat(sim.make_plane_waves([0.0]), 12)
    iq = usx.demodulate(sim.simulate(seq, ph), usx.DemodConfig())
    grid = usx.ScanGrid.cartesian(-0.003, 0.003, 21, 0.023, 0.027, 61)
    frames = usx.Beamformer().beamform_sequence(iq, grid, 1)
    cfg = usx.DopplerConfig()
    cfg.wall_filter = usx.WallFilter.NONE
    flow = usx.color_flow(frames, acq.prf, medium.speed_of_sound, cfg)
    valid = np.asarray(flow.valid).astype(bool)
    v = np.asarray(flow.velocity)[valid]
    assert abs(np.median(v)) < 0.02, "purely lateral flow must not register as axial velocity"


def test_scan_conversion_shape():
    grid = usx.ScanGrid.sector(-0.6, 0.6, 48, 0.01, 0.08, 96, usx.Vec3(0, 0, 0))
    img = np.ones((48, 96), dtype=np.float32)
    sc = usx.scan_convert(img, grid, 160, 200, -1.0)
    px = np.asarray(sc.pixels)
    assert px.shape == (200, 160)
    assert (px > 0.99).sum() > 3000, "the sector is filled"
    assert (px < -0.5).sum() > 1000, "the corners outside the sector are left as fill"


def test_pipeline_reuses_a_smoothed_reference(setup):
    probe, medium, acq, sim = setup
    ph = usx.phantoms.wire_targets(depths=(0.025,), lateral=(0.0,))
    cd = sim.simulate(sim.make_plane_waves([0.0]), ph)
    grid = usx.grid_for(probe, 0.015, 0.035, 48, 120)
    pipe = usx.ImagingPipeline(grid)
    a = np.asarray(pipe.process_display(cd))
    b = np.asarray(pipe.process_display(cd))
    assert pipe.frame_count == 2
    assert pipe.stats.total_ms > 0
    # Identical input, so the smoothed reference has converged and the second
    # frame must not be brighter or darker than the first.
    assert np.allclose(a, b, atol=1e-6)
    pipe.reset()
    assert pipe.frame_count == 0
