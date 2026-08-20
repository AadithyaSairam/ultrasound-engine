"""Probe and acquisition presets.

The numbers here are representative of real transducer classes rather than
copied from any one manufacturer's datasheet. They exist so that experiments
start from a physically sensible operating point: pitch near a wavelength for a
linear array and near half a wavelength for a phased array, sample rates at four
or more times the centre frequency, and a PRF whose unambiguous range actually
covers the depth being imaged.
"""

from __future__ import annotations

import numpy as np

from ._usx import Acquisition, Medium, Probe


def linear_5mhz() -> Probe:
    """General-purpose 5 MHz linear array — vascular, small parts, MSK."""
    p = Probe()
    p.n_elements = 128
    p.pitch = 300e-6
    p.element_width = 270e-6
    p.elevation_height = 5e-3
    p.center_frequency = 5.0e6
    p.bandwidth = 0.65
    return p


def linear_high_freq() -> Probe:
    """10 MHz fine-pitch linear array — superficial, high resolution, shallow."""
    p = Probe()
    p.n_elements = 128
    p.pitch = 150e-6
    p.element_width = 135e-6
    p.elevation_height = 3e-3
    p.center_frequency = 10.0e6
    p.bandwidth = 0.7
    return p


def phased_3mhz() -> Probe:
    """3 MHz phased array — cardiac. Half-wavelength pitch so it can steer a
    wide sector without grating lobes, and a small footprint to fit between
    ribs, which is why it images through a narrow window."""
    p = Probe()
    p.n_elements = 64
    p.pitch = 250e-6
    p.element_width = 220e-6
    p.elevation_height = 12e-3
    p.center_frequency = 3.0e6
    p.bandwidth = 0.6
    return p


def acquisition(probe: Probe, depth: float = 0.06, medium: Medium | None = None,
                samples_per_wavelength: float = 8.0, prf: float | None = None) -> Acquisition:
    """An acquisition sized to record echoes from the full depth.

    The record length is set by the round-trip time to `depth`, not guessed: too
    short and the bottom of the image is empty, too long and every stage pays for
    samples that contain nothing.
    """
    med = medium or Medium()
    c = med.speed_of_sound
    a = Acquisition()
    a.sampling_frequency = float(samples_per_wavelength * probe.center_frequency)
    round_trip = 2.0 * depth / c
    # Add a margin for the transmit delay spread across the aperture and the
    # pulse length, so the deepest echo is fully contained.
    margin = probe.aperture_width() / c + 4.0 / probe.center_frequency
    a.n_samples = int(np.ceil((round_trip + margin) * a.sampling_frequency))
    # Default PRF: as fast as the depth allows, so the next pulse is not fired
    # before the previous one's echoes have returned (which would fold deep
    # signal onto the top of the next image).
    a.prf = float(prf) if prf else float(0.95 * c / (2.0 * depth))
    a.t0 = 0.0
    return a


def plane_wave_sequence(sim, half_angle_deg: float = 9.0, n_angles: int = 11):
    """Symmetric steered plane waves.

    The useful angular span is roughly +/- arctan(1/(2*F#)) of the receive
    f-number; beyond that the added transmits contribute little new spatial
    frequency content and mostly add grating-lobe energy.
    """
    if n_angles <= 1:
        return sim.make_plane_waves([0.0])
    return sim.make_plane_waves(
        list(np.deg2rad(np.linspace(-half_angle_deg, half_angle_deg, n_angles)))
    )


def focused_sequence(sim, probe: Probe, n_lines: int = 128, focal_depth: float = 0.03):
    """Classic line-by-line focused scan across the aperture."""
    half = 0.5 * probe.aperture_width() * 0.9
    return sim.make_focused_scan(n_lines, -half, half, focal_depth)
