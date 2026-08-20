#include "usx/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace usx {
namespace {

// sinc(x) = sin(x)/x with the removable singularity handled.
inline real sinc(real x) {
  if (std::abs(x) < 1e-6f) return 1.0f - x * x / 6.0f;
  return std::sin(x) / x;
}

}  // namespace

void compute_transmit_delays(const Probe& probe, const Medium& medium,
                             Transmit& tx, Window apod_window,
                             real f_number, real* t0_offset) {
  const real c = medium.speed_of_sound;
  const int n = probe.n_elements;
  tx.delays.assign(n, 0.0f);
  tx.apodization.assign(n, 0.0f);

  const Vec3 O(tx.origin.x, 0.0f, 0.0f);  // beam origin on the transducer face

  // Half-width of the active transmit aperture. For a focused beam the
  // f-number sets it: F# = focal_depth / aperture, so aperture = depth / F#.
  // A plane wave uses the whole array — that is the entire point of it.
  real half_aperture;
  if (tx.type == TransmitType::Focused) {
    half_aperture = 0.5f * std::abs(tx.focus.z - O.z) / std::max(f_number, 0.1f);
  } else if (tx.type == TransmitType::Diverging) {
    half_aperture = 0.5f * probe.aperture_width();
  } else {
    half_aperture = 0.5f * probe.aperture_width();
  }
  half_aperture = std::min(half_aperture, 0.5f * probe.aperture_width());

  for (int i = 0; i < n; ++i) {
    const Vec3 p = probe.element_position(i);
    const real u = (p.x - O.x) / std::max(half_aperture, 1e-9f);
    const real w = window_value(apod_window, u);
    if (w <= 0.0f) continue;
    tx.apodization[i] = w;

    switch (tx.type) {
      case TransmitType::PlaneWave:
        // Flat wavefront with normal (sin a, 0, cos a). Element i must fire when
        // that plane, travelling at c, sweeps past it.
        tx.delays[i] = (p.x - O.x) * std::sin(tx.angle) / c;
        break;
      case TransmitType::Focused:
        // Every element's contribution must arrive at the focus at the same
        // instant, so the far elements (longer path) fire first.
        tx.delays[i] = (distance(tx.focus, O) - distance(tx.focus, p)) / c;
        break;
      case TransmitType::Diverging:
        // Time-reversed focusing: the array mimics a point source sitting
        // behind it, so the near elements fire first and the beam spreads.
        tx.delays[i] = (distance(p, tx.focus) - distance(O, tx.focus)) / c;
        break;
    }
  }

  // Delays above are in *engine time*, where t = 0 is the wavefront crossing the
  // beam origin — which means some of them are negative. Real transmit
  // electronics cannot fire before they start, so hardware fires element i at
  // (delays[i] - t0_offset) on its own clock, and the whole acquisition is
  // shifted by t0_offset. Exposing it here is what lets recorded data from a
  // real scanner be lined up with this engine's convention.
  real tmin = 0.0f;
  bool any = false;
  for (int i = 0; i < n; ++i) {
    if (tx.apodization[i] == 0.0f) continue;
    tmin = any ? std::min(tmin, tx.delays[i]) : tx.delays[i];
    any = true;
  }
  if (t0_offset) *t0_offset = any ? tmin : 0.0f;
}

real transmit_arrival_time(const Transmit& tx, const Vec3& point, real c) {
  const Vec3 O(tx.origin.x, 0.0f, 0.0f);
  switch (tx.type) {
    case TransmitType::PlaneWave: {
      // Projection of the point onto the wavefront normal.
      const real s = std::sin(tx.angle), k = std::cos(tx.angle);
      return ((point.x - O.x) * s + (point.z - O.z) * k) / c;
    }
    case TransmitType::Focused: {
      // Virtual-source model. Ahead of the focus the wavefront is converging,
      // so distance from the origin *decreases* as you approach it; past the
      // focus it diverges again. Getting this sign wrong produces an image that
      // is correctly focused in a band and mirrored/defocused beyond it.
      const Vec3 axis = tx.focus - O;
      const real axis_len = axis.norm();
      if (axis_len < 1e-9f) return point.norm() / c;
      const Vec3 d(axis.x / axis_len, axis.y / axis_len, axis.z / axis_len);
      const Vec3 rel = point - tx.focus;
      const real proj = rel.x * d.x + rel.y * d.y + rel.z * d.z;
      const real sign = (proj >= 0.0f) ? 1.0f : -1.0f;
      return (axis_len + sign * rel.norm()) / c;
    }
    case TransmitType::Diverging: {
      return (distance(point, tx.focus) - distance(O, tx.focus)) / c;
    }
  }
  return 0.0f;
}

real element_directivity(real element_width, real wavelength, real sin_theta) {
  // Far-field directivity of a rectangular piston in a soft baffle:
  //   D(theta) = cos(theta) * sinc(pi * w * sin(theta) / lambda)
  // The sinc is the Fourier transform of the element's uniform aperture; the
  // cosine is the obliquity factor. An element much narrower than a wavelength
  // is nearly omnidirectional, which is why fine-pitch arrays image wide
  // sectors and coarse ones do not.
  const real st = std::clamp(sin_theta, -1.0f, 1.0f);
  const real ct = std::sqrt(std::max(0.0f, 1.0f - st * st));
  return ct * sinc(kPi * element_width * st / wavelength);
}

real dynamic_aperture_weight(const Vec3& element, const Vec3& point,
                             real f_number, Window w, real tukey_alpha) {
  const real depth = point.z - element.z;
  if (depth <= 0.0f) return 0.0f;
  const real half_aperture = 0.5f * depth / std::max(f_number, 1e-3f);
  const real u = (element.x - point.x) / std::max(half_aperture, 1e-9f);
  return window_value(w, u, tukey_alpha);
}

}  // namespace usx
