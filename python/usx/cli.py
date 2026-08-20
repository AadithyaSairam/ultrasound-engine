"""Command line: `python -m usx <command>`.

Each subcommand runs one complete experiment end to end and writes images, so
that a change to the engine can be checked by looking at a picture rather than
only at a test result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import metrics, phantoms, presets, render
from ._usx import (Beamformer, BeamformerConfig, Combiner, Compounding, DemodConfig,
                   DopplerConfig, ImagingPipeline, Medium, PipelineConfig, ScanGrid,
                   Simulator, SimulatorConfig, Vec3, WallFilter, axial_strain, color_flow,
                   demodulate, envelope, estimate_axial_displacement, log_compress,
                   power_doppler)


def _grid(probe, z_min, z_max, nx, nz):
    half = 0.5 * probe.aperture_width()
    return ScanGrid.cartesian(-half, half, nx, z_min, z_max, nz)


def cmd_bmode(args):
    probe = getattr(presets, args.probe)()
    medium = Medium()
    acq = presets.acquisition(probe, depth=args.depth, medium=medium)
    sim = Simulator(probe, medium, acq, SimulatorConfig())

    print(f"phantom: building ...", flush=True)
    ph = phantoms.resolution_and_contrast(probe, medium, 0.005, args.depth)
    print(f"phantom: {len(ph)} scatterers")

    txs = presets.plane_wave_sequence(sim, args.half_angle, args.angles)
    t = time.perf_counter()
    raw = sim.simulate(txs, ph)
    print(f"simulate: {len(txs)} transmits in {time.perf_counter()-t:.1f} s")

    grid = _grid(probe, 0.005, args.depth, args.nx, args.nz)
    cfg = PipelineConfig()
    cfg.demod.decimation = 2
    pipe = ImagingPipeline(grid, cfg)
    img = pipe.process_display(raw)
    print(f"pipeline: {pipe.stats}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render.save_png(out, np.asarray(img).T, grid,
                    title=f"B-mode — {args.angles} plane waves, DAS")
    print(f"wrote {out}")

    env = np.asarray(envelope(pipe.process_frame(raw)))
    stats = metrics.summarize(env, grid, lesion_xz=(0.006, 0.030), lesion_radius=0.0035,
                              wire_xz=(-0.008, 0.024), speckle_xz=(0.0, 0.045))
    for k, v in stats.items():
        print(f"  {k:16s} {v: .3f}")


def cmd_compare(args):
    probe = getattr(presets, args.probe)()
    medium = Medium()
    acq = presets.acquisition(probe, depth=args.depth, medium=medium)
    sim = Simulator(probe, medium, acq, SimulatorConfig())
    ph = phantoms.resolution_and_contrast(probe, medium, 0.005, args.depth)
    print(f"phantom: {len(ph)} scatterers", flush=True)

    txs = presets.plane_wave_sequence(sim, args.half_angle, args.angles)
    raw = sim.simulate(txs, ph)
    iq = demodulate(raw, DemodConfig())
    grid = _grid(probe, 0.008, args.depth, args.nx, args.nz)

    panels, rows = [], []
    settings = [
        ("DAS", Combiner.DAS, False),
        ("DAS + CF", Combiner.DAS, True),
        ("DMAS", Combiner.DMAS, False),
        ("MV (L=16)", Combiner.MV, False),
    ]
    for label, comb, cf in settings:
        bc = BeamformerConfig()
        bc.f_number = 1.75
        bc.combiner = comb
        bc.coherence_factor = cf
        bc.mv_subarray = 16
        t = time.perf_counter()
        frame = Beamformer(bc).beamform(iq, grid)
        dt = time.perf_counter() - t
        env = np.asarray(envelope(frame))
        panels.append((f"{label}  ({dt*1e3:.0f} ms)",
                       np.asarray(log_compress(env, args.dynamic_range)).T))
        s = metrics.summarize(env, grid, lesion_xz=(0.006, 0.030), lesion_radius=0.0035,
                              wire_xz=(-0.008, 0.024), speckle_xz=(0.0, 0.045))
        s["label"] = label
        s["ms"] = dt * 1e3
        rows.append(s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render.save_comparison(out, panels, grid,
                           suptitle=f"Beamformer comparison — {args.angles} plane waves, "
                                    f"{args.dynamic_range:.0f} dB")
    print(f"wrote {out}\n")

    keys = ["label", "ms", "lateral_fwhm_mm", "axial_fwhm_mm", "sidelobe_db",
            "contrast_db", "cnr", "gcnr", "speckle_snr"]
    print("  ".join(f"{k:>16s}" for k in keys))
    for r in rows:
        print("  ".join(f"{r[k]:>16.3f}" if isinstance(r[k], float) else f"{r[k]:>16s}"
                        for k in keys))
    print(f"\n  (fully developed speckle has SNR {metrics.RAYLEIGH_SPECKLE_SNR:.2f}; "
          f"a much higher value means the texture has been smoothed away)")


def cmd_doppler(args):
    probe = presets.linear_5mhz()
    medium = Medium()
    acq = presets.acquisition(probe, depth=0.045, medium=medium, prf=args.prf)
    scfg = SimulatorConfig()
    # Blood scatters about 30 dB below tissue, so a flow acquisition sits far
    # closer to the receive noise floor than a B-mode one does. Real scanners buy
    # that margin back with higher transmit voltage, longer ensembles and
    # compounded packets; here it is simply set, and `--noise` exposes the knob
    # so the degradation can be watched.
    scfg.noise_db = args.noise
    sim = Simulator(probe, medium, acq, scfg)
    ph = phantoms.flow_phantom(probe, medium, vessel_depth=0.025, vessel_radius=0.004,
                               angle_deg=args.angle, peak_velocity=args.velocity)
    print(f"phantom: {len(ph)} scatterers, PRF {acq.prf:.0f} Hz", flush=True)

    base = sim.make_plane_waves([0.0])
    seq = Simulator.repeat(base, args.ensemble)
    t = time.perf_counter()
    raw = sim.simulate(seq, ph)
    print(f"simulate: {len(seq)} transmits in {time.perf_counter()-t:.1f} s")

    iq = demodulate(raw, DemodConfig())
    grid = _grid(probe, 0.008, 0.045, args.nx, args.nz)
    frames = Beamformer(BeamformerConfig()).beamform_sequence(iq, grid, len(base))

    dc = DopplerConfig()
    dc.wall_filter = WallFilter.POLYNOMIAL
    dc.polynomial_order = 1
    dc.kernel_axial = 6
    dc.kernel_lateral = 1
    flow = color_flow(frames, acq.prf, medium.speed_of_sound, dc)

    bmode = render.bmode_image(frames[0], args.dynamic_range)
    rgb = render.color_overlay(bmode, flow)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render.save_png(out, rgb, grid,
                    title=f"Colour flow — {args.ensemble}-packet ensemble, "
                          f"v_nyq = {flow.nyquist_velocity*100:.0f} cm/s")
    print(f"wrote {out}")

    pw = np.asarray(power_doppler(frames, WallFilter.POLYNOMIAL, 1))
    pw_db = 10 * np.log10(pw / (pw.max() + 1e-30) + 1e-12)
    render.save_png(out.with_name(out.stem + "_power.png"), pw_db.T, grid,
                    cmap="inferno", vmin=-40, vmax=0, colorbar_label="power (dB)",
                    title="Power Doppler")
    print(f"wrote {out.with_name(out.stem + '_power.png')}")

    v = np.asarray(flow.velocity)
    valid = np.asarray(flow.valid).astype(bool)
    if valid.any():
        expected = args.velocity * np.sin(np.deg2rad(args.angle))
        # The 99th percentile rather than the maximum: a single noisy pixel can
        # take any value up to the Nyquist limit, so quoting the max measures
        # the outlier, not the flow.
        peak = np.percentile(np.abs(v[valid]), 99)
        print(f"  coloured pixels          {valid.mean()*100:6.1f} % of the frame")
        print(f"  99th pct |v_axial|       {peak*100:6.1f} cm/s")
        print(f"  expected peak v_axial    {abs(expected)*100:6.1f} cm/s "
              f"(= {args.velocity*100:.0f} cm/s x sin {args.angle:.0f} deg)")
        print(f"  Nyquist limit            {flow.nyquist_velocity*100:6.1f} cm/s")


def cmd_strain(args):
    probe = presets.linear_5mhz()
    medium = Medium()
    acq = presets.acquisition(probe, depth=0.055, medium=medium)
    cfg = SimulatorConfig()
    cfg.enable_motion = False
    sim = Simulator(probe, medium, acq, cfg)

    inclusion = (0.0, 0.030, 0.005)
    before = phantoms.elastography_phantom(probe, medium, inclusion, args.ratio, 0.0)
    after = phantoms.elastography_phantom(probe, medium, inclusion, args.ratio, -args.strain)
    print(f"phantom: {len(before)} scatterers, {args.strain*100:.1f}% compression, "
          f"stiffness ratio {args.ratio:g}", flush=True)

    txs = presets.plane_wave_sequence(sim, 9.0, 5)
    iq_a = demodulate(sim.simulate(txs, before), DemodConfig())
    iq_b = demodulate(sim.simulate(txs, after), DemodConfig())

    grid = _grid(probe, 0.012, 0.050, args.nx, args.nz)
    bf = Beamformer(BeamformerConfig())
    fa, fb = bf.beamform(iq_a, grid), bf.beamform(iq_b, grid)

    disp = estimate_axial_displacement(fa, fb, medium.speed_of_sound, 10, 3,
                                       max_lag=args.max_lag)
    strain = np.asarray(axial_strain(disp, 25))
    corr = np.asarray(disp.correlation)
    strain = np.where(corr > 0.7, strain, np.nan)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render.save_png(out, np.asarray(render.bmode_image(fa, 60)), grid, title="B-mode (reference)")
    lim = np.nanpercentile(np.abs(strain), 98) or args.strain
    render.save_png(out.with_name(out.stem + "_strain.png"), strain.T * 100, grid,
                    cmap="viridis", vmin=-lim * 100, vmax=0, colorbar_label="axial strain (%)",
                    title=f"Strain elastogram — stiff inclusion at 30 mm")
    print(f"wrote {out} and {out.with_name(out.stem + '_strain.png')}")

    ax0, ax1 = np.asarray(grid.axis0), np.asarray(grid.axis1)
    X, Z = np.meshgrid(ax0, ax1, indexing="ij")
    inside = ((X - inclusion[0]) ** 2 + (Z - inclusion[1]) ** 2) < (inclusion[2] * 0.6) ** 2
    outside = (((X - inclusion[0]) ** 2 + (Z - inclusion[1]) ** 2) > (inclusion[2] * 1.8) ** 2) \
        & (np.abs(Z - inclusion[1]) < 0.010)
    def _mean(mask):
        vals = strain[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.abs(vals).mean()) if vals.size else float("nan")

    si, so = _mean(inside), _mean(outside)
    tracked = np.isfinite(strain).mean()
    print(f"  pixels tracked (corr > 0.7)     {tracked*100:6.1f} %")
    print(f"  mean |strain| inside inclusion  {si*100:6.3f} %")
    print(f"  mean |strain| in background     {so*100:6.3f} %")
    print(f"  measured ratio                  {so/max(si,1e-12):6.2f}  (applied {args.ratio:g})")


def cmd_bench(args):
    probe = presets.linear_5mhz()
    medium = Medium()
    acq = presets.acquisition(probe, depth=0.055, medium=medium)
    sim = Simulator(probe, medium, acq, SimulatorConfig())
    ph = phantoms.cyst_phantom(probe, medium)
    txs = presets.plane_wave_sequence(sim, 9.0, args.angles)
    print(f"simulating {len(txs)} transmits x {probe.n_elements} channels "
          f"x {acq.n_samples} samples ...", flush=True)
    raw = sim.simulate(txs, ph)

    grid = _grid(probe, 0.005, 0.055, args.nx, args.nz)
    cfg = PipelineConfig()
    cfg.demod.decimation = 2
    pipe = ImagingPipeline(grid, cfg)
    pipe.process_display(raw)  # warm up

    n = args.reps
    t = time.perf_counter()
    for _ in range(n):
        pipe.process_display(raw)
    dt = (time.perf_counter() - t) / n
    print(f"\n  grid            {grid.n_axis0} x {grid.n_axis1} px")
    print(f"  transmits       {len(txs)}")
    print(f"  last frame      {pipe.stats}")
    print(f"  mean end-to-end {dt*1e3:.1f} ms  ->  {1/dt:.1f} fps")


def main(argv=None):
    p = argparse.ArgumentParser(prog="usx", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bmode", help="simulate and beamform a B-mode image")
    b.add_argument("--probe", default="linear_5mhz")
    b.add_argument("--depth", type=float, default=0.055)
    b.add_argument("--angles", type=int, default=11)
    b.add_argument("--half-angle", type=float, default=9.0)
    b.add_argument("--nx", type=int, default=256)
    b.add_argument("--nz", type=int, default=640)
    b.add_argument("--out", default="out/bmode.png")
    b.set_defaults(func=cmd_bmode)

    c = sub.add_parser("compare", help="compare DAS / CF / DMAS / MV on one acquisition")
    c.add_argument("--probe", default="linear_5mhz")
    c.add_argument("--depth", type=float, default=0.050)
    c.add_argument("--angles", type=int, default=11)
    c.add_argument("--half-angle", type=float, default=9.0)
    c.add_argument("--nx", type=int, default=192)
    c.add_argument("--nz", type=int, default=480)
    c.add_argument("--dynamic-range", type=float, default=55.0)
    c.add_argument("--out", default="out/compare.png")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("doppler", help="colour and power Doppler on a flow phantom")
    d.add_argument("--ensemble", type=int, default=16)
    d.add_argument("--velocity", type=float, default=0.35)
    d.add_argument("--angle", type=float, default=20.0)
    d.add_argument("--prf", type=float, default=6000.0)
    d.add_argument("--noise", type=float, default=-80.0,
                   help="receive noise, dB below the peak sample")
    d.add_argument("--nx", type=int, default=160)
    d.add_argument("--nz", type=int, default=400)
    d.add_argument("--dynamic-range", type=float, default=50.0)
    d.add_argument("--out", default="out/doppler.png")
    d.set_defaults(func=cmd_doppler)

    s = sub.add_parser("strain", help="strain elastography on a stiff inclusion")
    s.add_argument("--strain", type=float, default=0.01)
    s.add_argument("--ratio", type=float, default=4.0)
    s.add_argument("--max-lag", type=int, default=12,
                   help="integer-sample search range for the coarse stage")
    s.add_argument("--nx", type=int, default=160)
    s.add_argument("--nz", type=int, default=480)
    s.add_argument("--out", default="out/strain.png")
    s.set_defaults(func=cmd_strain)

    bn = sub.add_parser("bench", help="time the pipeline")
    bn.add_argument("--angles", type=int, default=7)
    bn.add_argument("--nx", type=int, default=192)
    bn.add_argument("--nz", type=int, default=512)
    bn.add_argument("--reps", type=int, default=5)
    bn.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
