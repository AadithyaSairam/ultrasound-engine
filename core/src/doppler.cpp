#include "usx/doppler.hpp"

#include <algorithm>
#include <cmath>

namespace usx {
namespace {

// Build the (I - P) projector that removes the least-squares fit of a
// polynomial of the given order from a length-N slow-time signal.
//
// Constructing it once for the whole frame rather than per pixel is what makes
// polynomial regression filtering affordable: N is small (an ensemble is
// typically 8-32), so the N x N projector is tiny and every pixel reuses it.
std::vector<real> polynomial_reject_matrix(int n, int order) {
  order = std::clamp(order, 0, n - 1);
  const int p = order + 1;
  // Basis: Legendre-ish monomials on [-1, 1], orthonormalised by Gram-Schmidt.
  std::vector<std::vector<real>> basis;
  basis.reserve(static_cast<std::size_t>(p));
  for (int d = 0; d < p; ++d) {
    std::vector<real> v(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
      const real x = (n == 1) ? 0.0f : 2.0f * k / (n - 1) - 1.0f;
      v[static_cast<std::size_t>(k)] = std::pow(x, d);
    }
    for (const auto& b : basis) {
      real dot = 0;
      for (int k = 0; k < n; ++k) dot += v[static_cast<std::size_t>(k)] * b[static_cast<std::size_t>(k)];
      for (int k = 0; k < n; ++k) v[static_cast<std::size_t>(k)] -= dot * b[static_cast<std::size_t>(k)];
    }
    real norm = 0;
    for (real x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm < 1e-8f) continue;
    for (real& x : v) x /= norm;
    basis.push_back(std::move(v));
  }

  std::vector<real> M(static_cast<std::size_t>(n) * n, 0.0f);
  for (int i = 0; i < n; ++i) M[static_cast<std::size_t>(i) * n + i] = 1.0f;
  for (const auto& b : basis)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        M[static_cast<std::size_t>(i) * n + j] -= b[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(j)];
  return M;
}

}  // namespace

std::vector<Frame> apply_wall_filter(const std::vector<Frame>& ensemble,
                                     WallFilter type, int polynomial_order) {
  if (ensemble.empty()) return {};
  if (type == WallFilter::None) return ensemble;

  const int n = static_cast<int>(ensemble.size());
  const std::size_t n_pix = ensemble.front().data.size();
  for (const Frame& f : ensemble)
    if (f.data.size() != n_pix)
      throw std::invalid_argument("apply_wall_filter: frames have different sizes");

  const int order = (type == WallFilter::MeanSubtraction) ? 0 : polynomial_order;
  const std::vector<real> M = polynomial_reject_matrix(n, order);

  std::vector<Frame> out = ensemble;
  for (std::size_t k = 0; k < n_pix; ++k) {
    for (int i = 0; i < n; ++i) {
      cplx acc(0, 0);
      for (int j = 0; j < n; ++j)
        acc += ensemble[static_cast<std::size_t>(j)].data[k] * M[static_cast<std::size_t>(i) * n + j];
      out[static_cast<std::size_t>(i)].data[k] = acc;
    }
  }
  return out;
}

ColorFlowMap color_flow(const std::vector<Frame>& ensemble, real prf,
                        real speed_of_sound, DopplerConfig config) {
  if (ensemble.size() < 2)
    throw std::invalid_argument("color_flow: need at least 2 frames in the ensemble");

  const std::vector<Frame> filt =
      apply_wall_filter(ensemble, config.wall_filter, config.polynomial_order);

  const ScanGrid& grid = ensemble.front().grid;
  const int n0 = grid.n_axis0, n1 = grid.n_axis1;
  const int n_ens = static_cast<int>(filt.size());
  const real fdem = ensemble.front().demodulation_frequency;

  // Axial pixel spacing: the Loupas estimator measures phase per pixel, and this
  // is what converts that into a frequency.
  real dz = 1e-4f;
  if (n1 > 1) dz = std::abs(grid.axis1[1] - grid.axis1[0]);

  ColorFlowMap out;
  out.grid = grid;
  const std::size_t n_pix = static_cast<std::size_t>(n0) * n1;
  out.velocity.assign(n_pix, 0.0f);
  out.power.assign(n_pix, 0.0f);
  out.variance.assign(n_pix, 0.0f);
  out.valid.assign(n_pix, 0);
  out.nyquist_velocity = (fdem > 0.0f) ? speed_of_sound * prf / (4.0f * fdem) : 0.0f;

  const int ka = std::max(0, config.kernel_axial);
  const int kl = std::max(0, config.kernel_lateral);

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      const std::size_t k = static_cast<std::size_t>(i) * n1 + j;

      cplx R01(0, 0);   // slow-time lag 1: carries the Doppler shift
      cplx R10(0, 0);   // fast-time lag 1: carries the local centre frequency
      real R00 = 0.0f;  // zero-lag power

      for (int di = -kl; di <= kl; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= n0) continue;
        for (int dj = -ka; dj <= ka; ++dj) {
          const int jj = j + dj;
          if (jj < 0 || jj >= n1) continue;
          const std::size_t kk = static_cast<std::size_t>(ii) * n1 + jj;

          for (int e = 0; e < n_ens; ++e) {
            const cplx z = filt[static_cast<std::size_t>(e)].data[kk];
            R00 += std::norm(z);
            if (e + 1 < n_ens)
              R01 += std::conj(z) * filt[static_cast<std::size_t>(e + 1)].data[kk];
            if (jj + 1 < n1)
              R10 += std::conj(z) *
                     filt[static_cast<std::size_t>(e)].data[kk + 1];
          }
        }
      }

      out.power[k] = R00;
      if (R00 <= 0.0f) continue;

      // Normalized spectral width. |R01|/R00 near 1 means a narrow, coherent
      // spectrum (smooth laminar flow); near 0 means broad (turbulence, noise,
      // or just a bad estimate).
      const real coherence = std::abs(R01) / R00;
      out.variance[k] = std::clamp(2.0f * (1.0f - coherence), 0.0f, 1.0f);
      if (coherence < config.coherence_threshold) continue;

      const real phase_slow = std::arg(R01);
      real f_centre = fdem;
      if (config.loupas) {
        const real phase_fast = std::arg(R10);
        // Axial phase advance per pixel is 4*pi*f*dz/c, so this inverts to the
        // true local centre frequency.
        const real f_est = phase_fast * speed_of_sound / (4.0f * kPi * dz);
        // Guard against a degenerate estimate in noise-only regions.
        if (f_est > 0.25f * fdem && f_est < 2.0f * fdem) f_centre = f_est;
      }
      if (f_centre <= 0.0f) continue;

      // v = -arg(R01) * c * PRF / (4 * pi * f). The minus sign follows from the
      // IQ convention: increasing depth means a longer round trip, which the
      // e^{-j*2*pi*f*tau} mixer turns into a *decreasing* phase.
      out.velocity[k] = -phase_slow * speed_of_sound * prf / (4.0f * kPi * f_centre);
      out.valid[k] = 1;
    }
  }

  // Power thresholding: a velocity estimated from noise is a random number
  // uniformly distributed over the Nyquist range, and displaying it as flow is
  // how colour Doppler ends up painting speckle.
  real pmax = 0.0f;
  for (real p : out.power) pmax = std::max(pmax, p);
  if (pmax > 0.0f) {
    const real thr = pmax * std::pow(10.0f, config.power_threshold_db / 10.0f);
    for (std::size_t k = 0; k < n_pix; ++k)
      if (out.power[k] < thr) out.valid[k] = 0;
  }
  return out;
}

std::vector<real> power_doppler(const std::vector<Frame>& ensemble, WallFilter type,
                                int polynomial_order) {
  if (ensemble.empty()) return {};
  const std::vector<Frame> filt = apply_wall_filter(ensemble, type, polynomial_order);
  std::vector<real> p(filt.front().data.size(), 0.0f);
  for (const Frame& f : filt)
    for (std::size_t k = 0; k < p.size(); ++k) p[k] += std::norm(f.data[k]);
  return p;
}

DisplacementMap estimate_axial_displacement(const Frame& before, const Frame& after,
                                            real speed_of_sound,
                                            int kernel_axial, int kernel_lateral,
                                            int max_lag) {
  if (before.data.size() != after.data.size())
    throw std::invalid_argument("estimate_axial_displacement: frame size mismatch");

  const ScanGrid& grid = before.grid;
  const int n0 = grid.n_axis0, n1 = grid.n_axis1;
  real dz = 1e-4f;
  if (n1 > 1) dz = std::abs(grid.axis1[1] - grid.axis1[0]);
  const real fdem = before.demodulation_frequency;

  DisplacementMap out;
  out.grid = grid;
  const std::size_t n_pix = static_cast<std::size_t>(n0) * n1;
  out.axial.assign(n_pix, 0.0f);
  out.correlation.assign(n_pix, 0.0f);

  const int ka = std::max(1, kernel_axial);
  const int kl = std::max(0, kernel_lateral);
  const int lag_max = std::max(0, max_lag);

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      const std::size_t k = static_cast<std::size_t>(i) * n1 + j;

      // Local centre frequency from the reference frame's axial phase gradient,
      // so the phase-to-distance conversion does not have to trust f0.
      cplx R10(0, 0);
      for (int di = -kl; di <= kl; ++di) {
        const int ii = i + di;
        if (ii < 0 || ii >= n0) continue;
        for (int dj = -ka; dj <= ka; ++dj) {
          const int jj = j + dj;
          if (jj < 0 || jj + 1 >= n1) continue;
          const std::size_t kk = static_cast<std::size_t>(ii) * n1 + jj;
          R10 += std::conj(before.data[kk]) * before.data[kk + 1];
        }
      }
      real f_centre = fdem;
      const real f_est = std::arg(R10) * speed_of_sound / (4.0f * kPi * dz);
      if (f_est > 0.25f * fdem && f_est < 2.0f * fdem) f_centre = f_est;
      if (f_centre <= 0.0f) continue;

      // ---- Coarse stage: integer-sample lag with the highest correlation ----
      cplx best_R(0, 0);
      real best_corr = -1.0f;
      int best_lag = 0;
      for (int lag = -lag_max; lag <= lag_max; ++lag) {
        cplx R(0, 0);
        real ea = 0.0f, eb = 0.0f;
        for (int di = -kl; di <= kl; ++di) {
          const int ii = i + di;
          if (ii < 0 || ii >= n0) continue;
          for (int dj = -ka; dj <= ka; ++dj) {
            const int jj = j + dj;
            const int jl = jj + lag;
            if (jj < 0 || jj >= n1 || jl < 0 || jl >= n1) continue;
            const cplx a = before.data[static_cast<std::size_t>(ii) * n1 + jj];
            const cplx b = after.data[static_cast<std::size_t>(ii) * n1 + jl];
            R += std::conj(a) * b;
            ea += std::norm(a);
            eb += std::norm(b);
          }
        }
        if (ea <= 0.0f || eb <= 0.0f) continue;
        const real corr = std::abs(R) / std::sqrt(ea * eb);
        if (corr > best_corr) {
          best_corr = corr;
          best_R = R;
          best_lag = lag;
        }
      }
      if (best_corr < 0.0f) continue;

      out.correlation[k] = std::clamp(best_corr, 0.0f, 1.0f);
      // ---- Fine stage: the residual phase at the winning lag ----------------
      // Same conversion as Doppler, without the time axis: a displacement d
      // changes the round trip by 2d, hence the 4*pi.
      const real residual = -std::arg(best_R) * speed_of_sound / (4.0f * kPi * f_centre);
      out.axial[k] = static_cast<real>(best_lag) * dz + residual;
    }
  }
  return out;
}

std::vector<real> axial_strain(const DisplacementMap& disp, int fit_length) {
  const int n0 = disp.grid.n_axis0, n1 = disp.grid.n_axis1;
  real dz = 1e-4f;
  if (n1 > 1) dz = std::abs(disp.grid.axis1[1] - disp.grid.axis1[0]);

  const int half = std::max(1, fit_length / 2);
  std::vector<real> strain(static_cast<std::size_t>(n0) * n1, 0.0f);

  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      // Least-squares slope of displacement against depth over the window.
      // Sxx is a pure function of the window shape, so it is recomputed cheaply
      // rather than cached; the sum that matters is Sxy.
      real sx = 0, sy = 0, sxx = 0, sxy = 0;
      int n = 0;
      for (int d = -half; d <= half; ++d) {
        const int jj = j + d;
        if (jj < 0 || jj >= n1) continue;
        const real x = static_cast<real>(d) * dz;
        const real y = disp.axial[static_cast<std::size_t>(i) * n1 + jj];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
        ++n;
      }
      if (n < 2) continue;
      const real denom = n * sxx - sx * sx;
      if (std::abs(denom) < 1e-20f) continue;
      strain[static_cast<std::size_t>(i) * n1 + j] = (n * sxy - sx * sy) / denom;
    }
  }
  return strain;
}

}  // namespace usx
