#include "usx/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "usx/geometry.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace usx {
namespace {

// Gaussian envelope standard deviation that yields a given -6 dB fractional
// bandwidth at f0. The spectrum of exp(-t^2/(2*sigma^2))*cos(2*pi*f0*t) is a
// Gaussian of standard deviation 1/(2*pi*sigma) centred on f0; setting its
// half-amplitude full width to bandwidth*f0 gives this expression.
inline real gaussian_sigma_for_bandwidth(real f0, real bandwidth) {
  const real ln2 = 0.69314718f;
  return std::sqrt(2.0f * ln2) / (kPi * bandwidth * f0);
}

inline real db_to_amp(real db) { return std::pow(10.0f, db / 20.0f); }

}  // namespace

// ---------------------------------------------------------------------------
// Phantom factories
// ---------------------------------------------------------------------------

Phantom Phantom::point_targets(const std::vector<Vec3>& positions, real amplitude) {
  Phantom p;
  p.scatterers.reserve(positions.size());
  for (const Vec3& v : positions) {
    Scatterer s;
    s.position = v;
    s.amplitude = amplitude;
    p.scatterers.push_back(s);
  }
  return p;
}

Phantom Phantom::speckle_box(real x_min, real x_max, real z_min, real z_max,
                             real y_half_thickness, int n_scatterers,
                             std::uint32_t seed, real amplitude_sigma) {
  Phantom p;
  p.scatterers.reserve(static_cast<std::size_t>(std::max(0, n_scatterers)));
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real> ux(x_min, x_max);
  std::uniform_real_distribution<real> uz(z_min, z_max);
  std::uniform_real_distribution<real> uy(-y_half_thickness, y_half_thickness);
  // Rayleigh-distributed speckle arises from Gaussian-distributed scattering
  // strengths summed in a resolution cell, so the amplitudes are drawn normal.
  std::normal_distribution<real> amp(0.0f, amplitude_sigma);
  for (int i = 0; i < n_scatterers; ++i) {
    Scatterer s;
    s.position = Vec3(ux(rng), uy(rng), uz(rng));
    s.amplitude = amp(rng);
    p.scatterers.push_back(s);
  }
  return p;
}

int Phantom::recommended_scatterer_count(const Probe& probe, const Medium& medium,
                                         real x_span, real z_span, real y_span,
                                         real f_number, real per_cell) {
  const real lambda = probe.wavelength(medium.speed_of_sound);
  // Lateral cell ~ f_number * lambda; axial cell ~ pulse length ~ lambda / bw;
  // elevation cell is the (unfocused) slice thickness, approximated by the
  // elevation aperture.
  const real dx = std::max(f_number * lambda, 1e-5f);
  const real dz = std::max(lambda / std::max(probe.bandwidth, 0.1f), 1e-5f);
  const real dy = std::max(probe.elevation_height, 1e-4f);
  const real cells = (x_span / dx) * (z_span / dz) * std::max(y_span / dy, 1.0f);
  const double n = static_cast<double>(cells) * per_cell;
  return static_cast<int>(std::min(n, 4.0e6));
}

void Phantom::scale_amplitude_in_circle(const Vec3& center, real radius, real scale) {
  const real r2 = radius * radius;
  for (Scatterer& s : scatterers) {
    const real dx = s.position.x - center.x;
    const real dz = s.position.z - center.z;
    if (dx * dx + dz * dz <= r2) s.amplitude *= scale;
  }
}

void Phantom::add_flow_tube(const Vec3& center, real radius, real length, real axis_angle,
                            real peak_velocity, int n_scatterers, std::uint32_t seed,
                            real amplitude) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<real> along(-0.5f * length, 0.5f * length);
  std::uniform_real_distribution<real> u01(0.0f, 1.0f);
  std::normal_distribution<real> amp(0.0f, amplitude);

  const real ax = std::cos(axis_angle), az = std::sin(axis_angle);
  const real nx = -std::sin(axis_angle), nz = std::cos(axis_angle);

  for (int i = 0; i < n_scatterers; ++i) {
    // The vessel is a 3-D cylinder, so its cross-section must be filled
    // uniformly in *two* dimensions: the in-plane offset and the out-of-plane
    // (elevation) offset. Sampling only the in-plane offset — even correctly,
    // as R*sqrt(u) with a random sign — gives a line density proportional to
    // |r|, which is exactly zero on the vessel axis. That shows up in the image
    // as a dark stripe running down the middle of the vessel, which looks
    // enough like a real flow feature to be mistaken for one.
    const real rr = radius * std::sqrt(u01(rng));
    const real phi = 2.0f * kPi * u01(rng);
    const real r_off = rr * std::cos(phi);
    const real y_off = rr * std::sin(phi);
    const real s_along = along(rng);

    Scatterer s;
    s.position = Vec3(center.x + ax * s_along + nx * r_off, y_off,
                      center.z + az * s_along + nz * r_off);
    s.amplitude = amp(rng);
    // Poiseuille profile: v(r) = v_peak * (1 - (r/R)^2), with r the true radial
    // distance from the axis in three dimensions.
    const real v = peak_velocity * (1.0f - (rr / radius) * (rr / radius));
    s.velocity = Vec3(ax * v, 0.0f, az * v);
    scatterers.push_back(s);
  }
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------

Simulator::Simulator(Probe probe, Medium medium, Acquisition acq, SimulatorConfig config)
    : probe_(probe), medium_(medium), acq_(acq), config_(config) {
  probe_.validate();
  medium_.validate();
  acq_.validate();
  if (config_.oversampling < 1) config_.oversampling = 1;
}

std::vector<real> Simulator::pulse(real* half_duration) const {
  const real f0 = probe_.center_frequency;
  const real sigma = gaussian_sigma_for_bandwidth(f0, probe_.bandwidth);
  const real half = 3.0f * sigma;
  const real fs_os = acq_.sampling_frequency * config_.oversampling;
  const int n_half = static_cast<int>(std::ceil(half * fs_os));
  std::vector<real> p(static_cast<std::size_t>(2 * n_half + 1));
  for (int i = -n_half; i <= n_half; ++i) {
    const real t = static_cast<real>(i) / fs_os;
    p[static_cast<std::size_t>(i + n_half)] =
        std::exp(-0.5f * (t * t) / (sigma * sigma)) * std::cos(2.0f * kPi * f0 * t);
  }
  if (half_duration) *half_duration = static_cast<real>(n_half) / fs_os;
  return p;
}

std::vector<Transmit> Simulator::make_plane_waves(const std::vector<real>& angles) const {
  std::vector<Transmit> txs;
  txs.reserve(angles.size());
  for (real a : angles) {
    Transmit tx;
    tx.type = TransmitType::PlaneWave;
    tx.angle = a;
    tx.origin = Vec3(0, 0, 0);
    compute_transmit_delays(probe_, medium_, tx, config_.tx_window, config_.tx_f_number, nullptr);
    txs.push_back(std::move(tx));
  }
  return txs;
}

std::vector<Transmit> Simulator::make_focused_scan(int n_lines, real x_min, real x_max,
                                                   real focal_depth, real angle) const {
  std::vector<Transmit> txs;
  txs.reserve(static_cast<std::size_t>(std::max(0, n_lines)));
  for (int i = 0; i < n_lines; ++i) {
    const real x = (n_lines == 1) ? x_min
                                  : x_min + (x_max - x_min) * i / (n_lines - 1);
    Transmit tx;
    tx.type = TransmitType::Focused;
    tx.angle = angle;
    tx.origin = Vec3(x, 0, 0);
    tx.focus = Vec3(x + focal_depth * std::tan(angle), 0, focal_depth);
    compute_transmit_delays(probe_, medium_, tx, config_.tx_window, config_.tx_f_number, nullptr);
    txs.push_back(std::move(tx));
  }
  return txs;
}

std::vector<Transmit> Simulator::make_diverging_waves(const std::vector<real>& angles,
                                                      real virtual_source_depth) const {
  std::vector<Transmit> txs;
  txs.reserve(angles.size());
  for (real a : angles) {
    Transmit tx;
    tx.type = TransmitType::Diverging;
    tx.angle = a;
    tx.origin = Vec3(0, 0, 0);
    // Virtual source sits behind the array, displaced laterally to steer.
    tx.focus = Vec3(-virtual_source_depth * std::tan(a), 0, -virtual_source_depth);
    compute_transmit_delays(probe_, medium_, tx, config_.tx_window, config_.tx_f_number, nullptr);
    txs.push_back(std::move(tx));
  }
  return txs;
}

std::vector<Transmit> Simulator::repeat(const std::vector<Transmit>& txs, int n_ensemble) {
  std::vector<Transmit> out;
  out.reserve(txs.size() * static_cast<std::size_t>(std::max(1, n_ensemble)));
  for (int e = 0; e < n_ensemble; ++e)
    out.insert(out.end(), txs.begin(), txs.end());
  return out;
}

ChannelData Simulator::simulate(const std::vector<Transmit>& transmits,
                                const Phantom& phantom) const {
  ChannelData cd;
  cd.probe = probe_;
  cd.medium = medium_;
  cd.acq = acq_;
  cd.transmits = transmits;
  cd.is_iq = false;
  cd.allocate();

  const int n_ch = probe_.n_elements;
  const int n_smp = acq_.n_samples;
  const real c = medium_.speed_of_sound;
  const real fs = acq_.sampling_frequency;
  const int OS = config_.oversampling;
  const real fs_os = fs * OS;
  const real lambda = probe_.wavelength(c);

  real pulse_half = 0.0f;
  const std::vector<real> pulse_wave = pulse(&pulse_half);
  const int pulse_n = static_cast<int>(pulse_wave.size());

  // Attenuation in dB per metre at the centre frequency, one way.
  const real f0_mhz = probe_.center_frequency * 1e-6f;
  const real atten_db_per_m = config_.apply_attenuation
                                  ? medium_.attenuation * f0_mhz * 100.0f
                                  : 0.0f;

  // Precompute element positions once — this loop runs billions of times.
  std::vector<Vec3> elem(n_ch);
  for (int i = 0; i < n_ch; ++i) elem[i] = probe_.element_position(i);

  const real max_window_s = config_.max_incident_window_us * 1e-6f;
  const real trim_amp = db_to_amp(config_.incident_trim_db);

  for (int ev = 0; ev < static_cast<int>(transmits.size()); ++ev) {
    const Transmit& tx = transmits[ev];
    const real event_time = static_cast<real>(ev) / acq_.prf;

    real* out_event = cd.rf.data() + cd.index(ev, 0, 0);

    // Collect the elements that actually fire once, rather than testing the
    // apodization inside the hot loop.
    std::vector<int> active;
    active.reserve(static_cast<std::size_t>(n_ch));
    for (int i = 0; i < n_ch; ++i)
      if (!tx.apodization.empty() && tx.apodization[i] != 0.0f) active.push_back(i);
    if (active.empty())
      for (int i = 0; i < n_ch; ++i) active.push_back(i);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      // Each thread accumulates into a private copy of the event's channel
      // buffer and they are reduced at the end. Scatterers are independent, so
      // this parallelises cleanly; the alternative (atomics on every sample)
      // would be dominated by contention.
      std::vector<real> local(static_cast<std::size_t>(n_ch) * n_smp, 0.0f);
      std::vector<real> incident;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (std::size_t si = 0; si < phantom.scatterers.size(); ++si) {
        const Scatterer& sc = phantom.scatterers[si];
        if (sc.amplitude == 0.0f) continue;

        Vec3 pos = sc.position;
        if (config_.enable_motion) {
          pos = pos + sc.velocity * event_time;
        }
        if (pos.z <= 0.0f) continue;

        // ---- Stage 1: incident field at the scatterer --------------------
        // Sum the contribution of every firing element. This is what makes the
        // simulated transmit beam a real diffracted field (with a focal zone,
        // edge waves and sidelobes) rather than an idealized wavefront.
        real t_min = 0, t_max = 0;
        bool first = true;
        for (int i : active) {
          const real t = tx.delays[i] + distance(pos, elem[i]) / c;
          if (first) { t_min = t_max = t; first = false; }
          else { t_min = std::min(t_min, t); t_max = std::max(t_max, t); }
        }
        if (first) continue;

        real span = (t_max - t_min) + 2.0f * pulse_half;
        span = std::min(span, max_window_s + 2.0f * pulse_half);
        const real inc_t0 = t_min - pulse_half;
        const int inc_n = static_cast<int>(std::ceil(span * fs_os)) + 2;
        incident.assign(static_cast<std::size_t>(inc_n), 0.0f);

        const bool full = config_.full_transmit_diffraction;
        for (int i : active) {
          const Vec3& pe = elem[i];
          const real r = distance(pos, pe);
          if (r < 1e-6f) continue;
          real amp = tx.apodization.empty() ? 1.0f : tx.apodization[i];
          if (config_.apply_spreading) amp /= r;
          if (config_.apply_directivity) {
            const real sin_th = (pos.x - pe.x) / r;
            amp *= element_directivity(probe_.element_width, lambda, sin_th);
          }
          if (atten_db_per_m > 0.0f) amp *= db_to_amp(-atten_db_per_m * r);
          if (amp == 0.0f) continue;

          const real t_arr = tx.delays[i] + r / c;
          // Index of the pulse centre within the incident buffer.
          const real centre = (t_arr - inc_t0) * fs_os;
          const int base = static_cast<int>(std::floor(centre)) - pulse_n / 2;
          const real frac = centre - std::floor(centre);
          // Linear interpolation of the pulse onto the buffer grid. At 4x
          // oversampling of a 5 MHz pulse this is ~32 samples per cycle, where
          // linear interpolation error is well below the trim threshold.
          for (int k = 0; k < pulse_n; ++k) {
            const int idx = base + k;
            if (idx < 0 || idx + 1 >= inc_n) continue;
            const real v = amp * pulse_wave[static_cast<std::size_t>(k)];
            incident[static_cast<std::size_t>(idx)] += v * (1.0f - frac);
            incident[static_cast<std::size_t>(idx) + 1] += v * frac;
          }
          if (!full) break;  // geometric approximation: nearest element only
        }

        // Trim the low-level tails; everything downstream costs time
        // proportional to the length kept.
        real peak = 0.0f;
        for (real v : incident) peak = std::max(peak, std::abs(v));
        if (peak <= 0.0f) continue;
        const real thr = peak * trim_amp;
        int lo = 0, hi = inc_n - 1;
        while (lo < inc_n && std::abs(incident[static_cast<std::size_t>(lo)]) < thr) ++lo;
        while (hi > lo && std::abs(incident[static_cast<std::size_t>(hi)]) < thr) --hi;
        if (lo >= hi) continue;
        const real trim_t0 = inc_t0 + static_cast<real>(lo) / fs_os;
        const int trim_n = hi - lo + 1;

        // ---- Stage 2: echo received by each element ----------------------
        // The scatterer re-radiates the incident waveform; each element sees a
        // delayed, attenuated, direction-weighted copy of it.
        for (int j = 0; j < n_ch; ++j) {
          const Vec3& pe = elem[j];
          const real r = distance(pos, pe);
          if (r < 1e-6f) continue;
          real amp = sc.amplitude;
          if (config_.apply_spreading) amp /= r;
          if (config_.apply_directivity) {
            const real sin_th = (pos.x - pe.x) / r;
            amp *= element_directivity(probe_.element_width, lambda, sin_th);
          }
          if (atten_db_per_m > 0.0f) amp *= db_to_amp(-atten_db_per_m * r);
          if (amp == 0.0f) continue;

          const real delay = r / c;
          // Output sample range covered by the trimmed incident waveform.
          const real t_start = trim_t0 + delay;
          const real t_end = t_start + static_cast<real>(trim_n - 1) / fs_os;
          int n0 = static_cast<int>(std::ceil(acq_.time_to_sample(t_start)));
          int n1 = static_cast<int>(std::floor(acq_.time_to_sample(t_end)));
          n0 = std::max(n0, 0);
          n1 = std::min(n1, n_smp - 1);
          if (n0 > n1) continue;

          real* dst = local.data() + static_cast<std::size_t>(j) * n_smp;
          for (int n = n0; n <= n1; ++n) {
            const real t = acq_.sample_time(n) - delay;   // time in incident frame
            const real u = (t - trim_t0) * fs_os;
            const int k = static_cast<int>(u);
            if (k < 0 || k + 1 >= trim_n) continue;
            const real f = u - static_cast<real>(k);
            const real a = incident[static_cast<std::size_t>(lo + k)];
            const real b = incident[static_cast<std::size_t>(lo + k + 1)];
            dst[n] += amp * (a + (b - a) * f);
          }
        }
      }

#ifdef _OPENMP
#pragma omp critical
#endif
      {
        for (std::size_t k = 0; k < local.size(); ++k) out_event[k] += local[k];
      }
    }

    // ---- Receive noise ---------------------------------------------------
    if (config_.noise_db > -200.0f) {
      real peak = 0.0f;
      const std::size_t n_total = static_cast<std::size_t>(n_ch) * n_smp;
      for (std::size_t k = 0; k < n_total; ++k) peak = std::max(peak, std::abs(out_event[k]));
      if (peak > 0.0f) {
        std::mt19937 rng(config_.noise_seed + static_cast<std::uint32_t>(ev));
        std::normal_distribution<real> nd(0.0f, peak * db_to_amp(config_.noise_db));
        for (std::size_t k = 0; k < n_total; ++k) out_event[k] += nd(rng);
      }
    }
  }

  return cd;
}

}  // namespace usx
