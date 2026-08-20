#include "usx/types.hpp"

#include <algorithm>
#include <cmath>

namespace usx {

void Probe::validate() const {
  if (n_elements <= 0) throw std::invalid_argument("Probe.n_elements must be > 0");
  if (pitch <= 0) throw std::invalid_argument("Probe.pitch must be > 0");
  if (element_width <= 0) throw std::invalid_argument("Probe.element_width must be > 0");
  if (element_width > pitch * 1.001f)
    throw std::invalid_argument("Probe.element_width exceeds pitch (elements would overlap)");
  if (center_frequency <= 0) throw std::invalid_argument("Probe.center_frequency must be > 0");
  if (bandwidth <= 0) throw std::invalid_argument("Probe.bandwidth must be > 0");
}

void Medium::validate() const {
  if (speed_of_sound <= 0) throw std::invalid_argument("Medium.speed_of_sound must be > 0");
  if (attenuation < 0) throw std::invalid_argument("Medium.attenuation must be >= 0");
}

void Acquisition::validate() const {
  if (sampling_frequency <= 0) throw std::invalid_argument("Acquisition.sampling_frequency must be > 0");
  if (n_samples <= 0) throw std::invalid_argument("Acquisition.n_samples must be > 0");
  if (prf <= 0) throw std::invalid_argument("Acquisition.prf must be > 0");
}

int Transmit::n_active() const {
  int n = 0;
  for (real w : apodization)
    if (w != 0.0f) ++n;
  return n;
}

void ChannelData::allocate() {
  validate();
  if (is_iq) {
    iq.assign(size(), cplx(0, 0));
    rf.clear();
  } else {
    rf.assign(size(), 0.0f);
    iq.clear();
  }
}

void ChannelData::validate() const {
  probe.validate();
  medium.validate();
  acq.validate();
  if (transmits.empty()) throw std::invalid_argument("ChannelData has no transmit events");
  for (const Transmit& tx : transmits) {
    if (!tx.delays.empty() && static_cast<int>(tx.delays.size()) != probe.n_elements)
      throw std::invalid_argument("Transmit.delays length != probe.n_elements");
    if (!tx.apodization.empty() && static_cast<int>(tx.apodization.size()) != probe.n_elements)
      throw std::invalid_argument("Transmit.apodization length != probe.n_elements");
  }
  const std::size_t expect = size();
  const std::size_t have = is_iq ? iq.size() : rf.size();
  if (have != 0 && have != expect)
    throw std::invalid_argument("ChannelData buffer size does not match n_events*n_channels*n_samples");
  if (is_iq && demodulation_frequency <= 0)
    throw std::invalid_argument("IQ ChannelData needs a positive demodulation_frequency");
}

ScanGrid ScanGrid::cartesian(real x_min, real x_max, int nx,
                             real z_min, real z_max, int nz) {
  ScanGrid g;
  g.type = GridType::Cartesian;
  g.n_axis0 = nx;
  g.n_axis1 = nz;
  g.axis0.resize(nx);
  g.axis1.resize(nz);
  for (int i = 0; i < nx; ++i)
    g.axis0[i] = nx == 1 ? x_min : x_min + (x_max - x_min) * i / (nx - 1);
  for (int j = 0; j < nz; ++j)
    g.axis1[j] = nz == 1 ? z_min : z_min + (z_max - z_min) * j / (nz - 1);
  g.build_points();
  g.validate();
  return g;
}

ScanGrid ScanGrid::sector(real theta_min, real theta_max, int n_theta,
                          real r_min, real r_max, int n_r, Vec3 apex) {
  ScanGrid g;
  g.type = GridType::Sector;
  g.n_axis0 = n_theta;
  g.n_axis1 = n_r;
  g.apex = apex;
  g.axis0.resize(n_theta);
  g.axis1.resize(n_r);
  for (int i = 0; i < n_theta; ++i)
    g.axis0[i] = n_theta == 1 ? theta_min
                              : theta_min + (theta_max - theta_min) * i / (n_theta - 1);
  for (int j = 0; j < n_r; ++j)
    g.axis1[j] = n_r == 1 ? r_min : r_min + (r_max - r_min) * j / (n_r - 1);
  g.build_points();
  g.validate();
  return g;
}

void ScanGrid::build_points() {
  points.resize(static_cast<std::size_t>(n_axis0) * n_axis1);
  for (int i = 0; i < n_axis0; ++i) {
    for (int j = 0; j < n_axis1; ++j) {
      const std::size_t k = static_cast<std::size_t>(i) * n_axis1 + j;
      if (type == GridType::Cartesian) {
        points[k] = Vec3(axis0[i], 0.0f, axis1[j]);
      } else {
        const real th = axis0[i], r = axis1[j];
        points[k] = Vec3(apex.x + r * std::sin(th), 0.0f, apex.z + r * std::cos(th));
      }
    }
  }
}

void ScanGrid::validate() const {
  if (n_axis0 <= 0 || n_axis1 <= 0)
    throw std::invalid_argument("ScanGrid dimensions must be > 0");
  if (static_cast<int>(axis0.size()) != n_axis0 || static_cast<int>(axis1.size()) != n_axis1)
    throw std::invalid_argument("ScanGrid axis length mismatch");
  if (points.size() != static_cast<std::size_t>(n_axis0) * n_axis1)
    throw std::invalid_argument("ScanGrid points not built (call build_points)");
}

real window_value(Window w, real u, real tukey_alpha) {
  if (u < -1.0f || u > 1.0f) return 0.0f;
  const real t = 0.5f * (u + 1.0f);  // map to [0, 1]
  switch (w) {
    case Window::Rect:
      return 1.0f;
    case Window::Hann:
      return 0.5f * (1.0f - std::cos(2.0f * kPi * t));
    case Window::Hamming:
      return 0.54f - 0.46f * std::cos(2.0f * kPi * t);
    case Window::Blackman:
      return 0.42f - 0.5f * std::cos(2.0f * kPi * t) + 0.08f * std::cos(4.0f * kPi * t);
    case Window::Tukey: {
      const real a = std::clamp(tukey_alpha, 0.0f, 1.0f);
      if (a <= 0.0f) return 1.0f;
      if (t < a * 0.5f) return 0.5f * (1.0f + std::cos(2.0f * kPi / a * (t - a * 0.5f)));
      if (t > 1.0f - a * 0.5f)
        return 0.5f * (1.0f + std::cos(2.0f * kPi / a * (t - 1.0f + a * 0.5f)));
      return 1.0f;
    }
  }
  return 1.0f;
}

Window window_from_string(const std::string& name) {
  if (name == "rect" || name == "boxcar" || name == "none") return Window::Rect;
  if (name == "hann" || name == "hanning") return Window::Hann;
  if (name == "hamming") return Window::Hamming;
  if (name == "tukey") return Window::Tukey;
  if (name == "blackman") return Window::Blackman;
  throw std::invalid_argument("unknown window: " + name);
}

}  // namespace usx
