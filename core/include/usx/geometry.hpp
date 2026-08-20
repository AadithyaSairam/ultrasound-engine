// usx/geometry.hpp — Transmit delay laws and wavefront arrival times.
//
// This header is the single source of truth for "when does sound get there".
// Both the simulator (which needs to know when to emit) and the beamformer
// (which needs to know when to listen) call these functions. Keeping one copy
// is not just tidiness: transmit timing conventions are the most common source
// of a beamformed image that is subtly, uniformly out of focus, and a shared
// implementation means a geometry bug shows up as a *test* failure rather than
// as a slightly-blurry picture nobody notices.
#pragma once

#include "usx/types.hpp"

namespace usx {

// Build the per-element delay law for a transmit event.
//
// The delays returned are "fire times": element i is excited at t = delays[i],
// with the smallest delay normalized to 0 so that no element is asked to fire
// before the acquisition starts. `t0_offset` receives the time at which the
// resulting wavefront passes through the origin (0,0,0), which the caller
// subtracts to restore the engine's t = 0 convention.
void compute_transmit_delays(const Probe& probe, const Medium& medium,
                             Transmit& tx, Window apod_window,
                             real f_number, real* t0_offset);

// Time at which the *ideal* transmit wavefront reaches `point`, relative to the
// engine's t = 0 (wavefront crossing the origin).
//
// This is the model a beamformer uses. It is an idealization — the real field
// from a finite aperture has edge waves and diffraction that this expression
// ignores — which is exactly why the simulator does NOT use it, and instead
// sums the true per-element contributions. That asymmetry is deliberate: it
// means beamforming simulated data is a genuine test of the delay model, not a
// tautology. See docs/THEORY.md, "Why the simulator and beamformer disagree".
real transmit_arrival_time(const Transmit& tx, const Vec3& point, real c);

// Time for the echo to travel from `point` back to a receive element.
inline real receive_arrival_time(const Vec3& element, const Vec3& point, real c) {
  return distance(element, point) / c;
}

// Element directivity: the far-field amplitude response of a rectangular
// element of finite width, which falls off as sinc(pi * w * sin(theta) / lambda).
// A wide element is a more sensitive but more directional receiver — this is
// why sub-wavelength element widths matter for wide-angle imaging.
real element_directivity(real element_width, real wavelength, real sin_theta);

// f-number-limited dynamic receive aperture.
//
// Returns the receive apodization weight for `element` when imaging `point`.
// The f-number (depth / aperture width) is held constant with depth, so the
// aperture opens up as you image deeper. This keeps the lateral resolution
// cell a roughly constant number of wavelengths wide instead of degrading with
// depth, and suppresses the highly-oblique channels whose echoes are both weak
// and geometrically unreliable.
real dynamic_aperture_weight(const Vec3& element, const Vec3& point,
                             real f_number, Window w, real tukey_alpha = 0.5f);

}  // namespace usx
