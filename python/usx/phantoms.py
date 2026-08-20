"""Standard test phantoms.

These mirror the physical phantoms used to characterise real scanners: a wire
target for resolution, an anechoic cyst for contrast, a flow tube for Doppler,
and a stiff inclusion for elastography. Using the same targets means the numbers
this engine reports can be compared against published ones rather than only
against itself.
"""

from __future__ import annotations

import numpy as np

from ._usx import Medium, Phantom, Probe, Scatterer, Vec3


def _speckle(probe: Probe, medium: Medium, x_half: float, z_min: float, z_max: float,
             y_half: float, seed: int, density_per_cell: float) -> Phantom:
    n = Phantom.recommended_scatterer_count(
        probe, medium, 2 * x_half, z_max - z_min, 2 * y_half,
        f_number=2.0, per_cell=density_per_cell)
    n = int(min(max(n, 2000), 400_000))
    return Phantom.speckle_box(-x_half, x_half, z_min, z_max, y_half, n, seed)


def wire_targets(depths=(0.010, 0.020, 0.030, 0.040, 0.050),
                 lateral=(-0.006, 0.0, 0.006)) -> Phantom:
    """A grid of isolated point scatterers.

    The image of one of these is the point spread function, so this phantom is
    what resolution is measured on. Note the targets are deliberately far apart:
    two scatterers inside one resolution cell interfere, and the result measures
    their interference, not the system.
    """
    pts = [Vec3(float(x), 0.0, float(z)) for z in depths for x in lateral]
    return Phantom.point_targets(pts, 1.0)


def cyst_phantom(probe: Probe, medium: Medium | None = None,
                 cysts=((0.0, 0.020, 0.003, 0.0),
                        (-0.007, 0.035, 0.003, 0.25),
                        (0.007, 0.035, 0.003, 2.5)),
                 z_min: float = 0.005, z_max: float = 0.055,
                 seed: int = 3, density_per_cell: float = 12.0) -> Phantom:
    """Speckle background with anechoic, hypoechoic and hyperechoic inclusions.

    Each cyst is (x, z, radius, amplitude_scale). Scale 0 is anechoic (a fluid
    cyst), below 1 is hypoechoic, above 1 hyperechoic. This is the phantom CNR
    and gCNR are computed on.
    """
    med = medium or Medium()
    half = 0.5 * probe.aperture_width() * 1.15
    ph = _speckle(probe, med, half, z_min, z_max, 2e-4, seed, density_per_cell)
    for x, z, r, scale in cysts:
        ph.scale_amplitude_in_circle(Vec3(float(x), 0.0, float(z)), float(r), float(scale))
    return ph


def resolution_and_contrast(probe: Probe, medium: Medium | None = None,
                            z_min: float = 0.005, z_max: float = 0.055,
                            seed: int = 5) -> Phantom:
    """The combined phantom: wires for resolution plus a cyst for contrast.

    Real characterisation phantoms combine both because the two trade against
    each other — a beamformer can always buy contrast by blurring, and a wire
    target is what catches it doing so.
    """
    ph = cyst_phantom(probe, medium, cysts=((0.006, 0.030, 0.0035, 0.0),),
                      z_min=z_min, z_max=z_max, seed=seed)
    wires = Phantom.point_targets(
        [Vec3(-0.008, 0.0, float(z)) for z in (0.012, 0.024, 0.036, 0.048)], 40.0)
    ph.extend(wires)
    return ph


def flow_phantom(probe: Probe, medium: Medium | None = None,
                 vessel_depth: float = 0.025, vessel_radius: float = 0.004,
                 angle_deg: float = 20.0, peak_velocity: float = 0.35,
                 z_min: float = 0.005, z_max: float = 0.045,
                 seed: int = 7, with_tissue: bool = True) -> Phantom:
    """A vessel with laminar flow inside stationary tissue.

    `angle_deg` is the angle between the vessel and the transducer face. Doppler
    measures only the velocity component along the beam, so a vessel running
    exactly parallel to the face (0 degrees) produces no signal at all — which is
    why sonographers angle the probe, and why reported velocities are divided by
    cos(theta).
    """
    med = medium or Medium()
    ph = Phantom()
    if with_tissue:
        half = 0.5 * probe.aperture_width() * 1.15
        ph = _speckle(probe, med, half, z_min, z_max, 2e-4, seed, 10.0)
        ph.scale_amplitude_in_circle(Vec3(0.0, 0.0, vessel_depth), vessel_radius, 0.0)
    ph.add_flow_tube(Vec3(0.0, 0.0, vessel_depth), vessel_radius, 0.06,
                     float(np.deg2rad(angle_deg)), peak_velocity, 20000, seed + 1, 0.03)
    return ph


def elastography_phantom(probe: Probe, medium: Medium | None = None,
                         inclusion=(0.0, 0.030, 0.005), stiffness_ratio: float = 4.0,
                         strain: float = 0.0, z_min: float = 0.010, z_max: float = 0.050,
                         seed: int = 13) -> Phantom:
    """A soft background containing a stiff inclusion, optionally compressed.

    `strain` is the bulk axial strain applied from the top of the imaged region;
    pass 0 for the reference (uncompressed) state and a small negative number
    (say -0.01, i.e. 1% compression) for the deformed one. Estimating the
    displacement between the two and differentiating it gives the strain image,
    in which the stiff inclusion appears dark because it strains less.

    Inside the inclusion the local strain is reduced by `stiffness_ratio`. The
    displacement field is obtained by *integrating* that local strain down each
    axial line, so it stays continuous across the boundary: a discontinuous
    displacement is physically impossible and would produce a strain image of
    the seam rather than of the inclusion.
    """
    med = medium or Medium()
    half = 0.5 * probe.aperture_width() * 1.15
    ph = _speckle(probe, med, half, z_min, z_max, 2e-4, seed, 12.0)
    if strain == 0.0:
        return ph

    cx, cz, r = inclusion
    pos = np.asarray(ph.positions, dtype=np.float64)
    x, z = pos[:, 0], pos[:, 2]

    # Length of the inclusion actually traversed on the way down to depth z.
    dx = x - cx
    chord = 2.0 * np.sqrt(np.clip(r * r - dx * dx, 0.0, None))
    z_entry = cz - 0.5 * chord
    traversed = np.clip(z - z_entry, 0.0, chord)

    e_soft = float(strain)
    e_stiff = e_soft / float(stiffness_ratio)
    disp = e_soft * (z - z_min) + (e_stiff - e_soft) * traversed

    out = ph.copy()
    moved = pos.copy()
    moved[:, 2] = z + disp
    out.positions = moved.astype(np.float32)
    out.amplitudes = np.asarray(ph.amplitudes, dtype=np.float32)
    return out
