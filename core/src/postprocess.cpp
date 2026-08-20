#include "usx/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace usx {
namespace {

// Iterative radix-2 FFT. Small, dependency-free, and only used for the Hilbert
// transform, where the transform length is a few hundred to a few thousand.
void fft(std::vector<cplx>& a, bool inverse) {
  const std::size_t n = a.size();
  if (n < 2) return;
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const real ang = 2.0f * kPi / static_cast<real>(len) * (inverse ? 1.0f : -1.0f);
    const cplx wl(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      cplx w(1, 0);
      for (std::size_t k = 0; k < len / 2; ++k) {
        const cplx u = a[i + k];
        const cplx v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
        w *= wl;
      }
    }
  }
  if (inverse)
    for (cplx& v : a) v /= static_cast<real>(n);
}

std::size_t next_pow2(std::size_t n) {
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

}  // namespace

std::vector<cplx> analytic_along_axis1(const std::vector<real>& data, int n_axis0, int n_axis1) {
  std::vector<cplx> out(static_cast<std::size_t>(n_axis0) * n_axis1);
  const std::size_t n_fft = next_pow2(static_cast<std::size_t>(n_axis1));
  std::vector<cplx> buf(n_fft);
  for (int i = 0; i < n_axis0; ++i) {
    std::fill(buf.begin(), buf.end(), cplx(0, 0));
    for (int j = 0; j < n_axis1; ++j)
      buf[static_cast<std::size_t>(j)] = cplx(data[static_cast<std::size_t>(i) * n_axis1 + j], 0.0f);
    fft(buf, false);
    // Zero the negative frequencies and double the positive ones: the standard
    // construction of the analytic signal, whose modulus is the envelope.
    for (std::size_t k = 1; k < n_fft / 2; ++k) buf[k] *= 2.0f;
    for (std::size_t k = n_fft / 2 + 1; k < n_fft; ++k) buf[k] = cplx(0, 0);
    fft(buf, true);
    for (int j = 0; j < n_axis1; ++j)
      out[static_cast<std::size_t>(i) * n_axis1 + j] = buf[static_cast<std::size_t>(j)];
  }
  return out;
}

std::vector<real> envelope(const Frame& frame) {
  const int n0 = frame.grid.n_axis0, n1 = frame.grid.n_axis1;
  std::vector<real> env(static_cast<std::size_t>(n0) * n1);

  if (frame.demodulation_frequency > 0.0f) {
    for (std::size_t k = 0; k < env.size(); ++k) env[k] = std::abs(frame.data[k]);
    return env;
  }
  // RF-beamformed frame: the imaginary part is empty and the real part still
  // carries the carrier, so build the analytic signal first.
  std::vector<real> re(static_cast<std::size_t>(n0) * n1);
  for (std::size_t k = 0; k < re.size(); ++k) re[k] = frame.data[k].real();
  const std::vector<cplx> an = analytic_along_axis1(re, n0, n1);
  for (std::size_t k = 0; k < env.size(); ++k) env[k] = std::abs(an[k]);
  return env;
}

std::vector<real> apply_tgc(const std::vector<real>& env, const ScanGrid& grid,
                            real attenuation_db_cm_mhz, real center_frequency_hz) {
  std::vector<real> out(env.size());
  const real f_mhz = center_frequency_hz * 1e-6f;
  for (int i = 0; i < grid.n_axis0; ++i) {
    for (int j = 0; j < grid.n_axis1; ++j) {
      const std::size_t k = static_cast<std::size_t>(i) * grid.n_axis1 + j;
      const real depth_cm = grid.points[k].z * 100.0f;
      // Round trip, hence the factor of 2.
      const real gain_db = 2.0f * attenuation_db_cm_mhz * f_mhz * std::max(depth_cm, 0.0f);
      out[k] = env[k] * std::pow(10.0f, gain_db / 20.0f);
    }
  }
  return out;
}

std::vector<real> log_compress(const std::vector<real>& env, real dynamic_range_db,
                               real gain_db, real reference) {
  real ref = reference;
  if (ref <= 0.0f) {
    ref = 0.0f;
    for (real v : env) ref = std::max(ref, v);
  }
  if (ref <= 0.0f) return std::vector<real>(env.size(), 0.0f);

  std::vector<real> out(env.size());
  const real dr = std::max(dynamic_range_db, 1.0f);
  for (std::size_t k = 0; k < env.size(); ++k) {
    const real v = env[k] / ref;
    const real db = (v > 0.0f) ? 20.0f * std::log10(v) : -1e6f;
    out[k] = std::clamp((db + gain_db + dr) / dr, 0.0f, 1.0f);
  }
  return out;
}

ScanConverted scan_convert(const std::vector<real>& image, const ScanGrid& grid,
                           int width, int height, real fill_value) {
  ScanConverted sc;
  sc.width = width;
  sc.height = height;
  sc.pixels.assign(static_cast<std::size_t>(width) * height, fill_value);

  // Physical extent covered by the grid's sample points.
  real x_min = std::numeric_limits<real>::max(), x_max = -x_min;
  real z_min = x_min, z_max = -x_min;
  for (const Vec3& p : grid.points) {
    x_min = std::min(x_min, p.x); x_max = std::max(x_max, p.x);
    z_min = std::min(z_min, p.z); z_max = std::max(z_max, p.z);
  }
  sc.x_min = x_min; sc.x_max = x_max; sc.z_min = z_min; sc.z_max = z_max;

  const bool cartesian = grid.type == GridType::Cartesian;
  const real th0 = grid.axis0.front(), th1 = grid.axis0.back();
  const real r0 = grid.axis1.front(), r1 = grid.axis1.back();

  for (int py = 0; py < height; ++py) {
    const real z = z_min + (z_max - z_min) * (height == 1 ? 0.0f : static_cast<real>(py) / (height - 1));
    for (int px = 0; px < width; ++px) {
      const real x = x_min + (x_max - x_min) * (width == 1 ? 0.0f : static_cast<real>(px) / (width - 1));

      // Invert the grid's own mapping to find where this display pixel sits in
      // acquisition coordinates, then bilinearly interpolate. Doing it this way
      // (backward mapping) guarantees every output pixel is written exactly
      // once, unlike splatting each sample forward, which leaves holes.
      real u, v;
      if (cartesian) {
        u = (x - grid.axis0.front()) / std::max(grid.axis0.back() - grid.axis0.front(), 1e-9f) *
            (grid.n_axis0 - 1);
        v = (z - grid.axis1.front()) / std::max(grid.axis1.back() - grid.axis1.front(), 1e-9f) *
            (grid.n_axis1 - 1);
      } else {
        const real dx = x - grid.apex.x, dz = z - grid.apex.z;
        const real r = std::sqrt(dx * dx + dz * dz);
        const real th = std::atan2(dx, dz);
        if (r < std::min(r0, r1) || r > std::max(r0, r1) ||
            th < std::min(th0, th1) || th > std::max(th0, th1))
          continue;
        u = (th - th0) / std::max(th1 - th0, 1e-9f) * (grid.n_axis0 - 1);
        v = (r - r0) / std::max(r1 - r0, 1e-9f) * (grid.n_axis1 - 1);
      }

      const int i0 = static_cast<int>(std::floor(u)), j0 = static_cast<int>(std::floor(v));
      if (i0 < 0 || j0 < 0 || i0 + 1 >= grid.n_axis0 || j0 + 1 >= grid.n_axis1) continue;
      const real fu = u - i0, fv = v - j0;
      const std::size_t n1 = static_cast<std::size_t>(grid.n_axis1);
      const real a = image[static_cast<std::size_t>(i0) * n1 + j0];
      const real b = image[static_cast<std::size_t>(i0 + 1) * n1 + j0];
      const real cc = image[static_cast<std::size_t>(i0) * n1 + j0 + 1];
      const real d = image[static_cast<std::size_t>(i0 + 1) * n1 + j0 + 1];
      sc.pixels[static_cast<std::size_t>(py) * width + px] =
          (a * (1 - fu) + b * fu) * (1 - fv) + (cc * (1 - fu) + d * fu) * fv;
    }
  }
  return sc;
}

}  // namespace usx
