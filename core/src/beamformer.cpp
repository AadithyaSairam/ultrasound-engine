#include "usx/beamformer.hpp"

#include <algorithm>
#include <cmath>

#include "usx/geometry.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace usx {
namespace {

inline void fast_sincos(real x, real* s, real* c) {
#if defined(__GNUC__) && !defined(__clang__)
  sincosf(x, s, c);
#else
  *s = std::sin(x);
  *c = std::cos(x);
#endif
}

// Linear interpolation of a complex trace at a fractional sample index.
// Legitimate on baseband IQ, where the carrier has been removed and what is
// left varies on the scale of the pulse envelope rather than the wavelength.
inline cplx interp(const cplx* trace, int n, real idx) {
  const int k = static_cast<int>(std::floor(idx));
  if (k < 0 || k + 1 >= n) return cplx(0, 0);
  const real f = idx - static_cast<real>(k);
  return trace[k] * (1.0f - f) + trace[k + 1] * f;
}

inline cplx interp_rf(const real* trace, int n, real idx) {
  const int k = static_cast<int>(std::floor(idx));
  if (k < 0 || k + 1 >= n) return cplx(0, 0);
  const real f = idx - static_cast<real>(k);
  return cplx(trace[k] * (1.0f - f) + trace[k + 1] * f, 0.0f);
}

// In-place Cholesky factorisation of a Hermitian positive-definite matrix,
// followed by a solve. Written out rather than pulled from a linear algebra
// library because the matrices here are tiny (L <= 32) and called once per
// pixel: the call overhead of a general BLAS would dominate the arithmetic.
// Returns false if the matrix is not positive definite, which is the signal
// that diagonal loading was insufficient.
bool cholesky_solve(std::vector<cplx>& A, int L, std::vector<cplx>& b) {
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j <= i; ++j) {
      cplx sum = A[static_cast<std::size_t>(i) * L + j];
      for (int k = 0; k < j; ++k)
        sum -= A[static_cast<std::size_t>(i) * L + k] *
               std::conj(A[static_cast<std::size_t>(j) * L + k]);
      if (i == j) {
        const real d = sum.real();
        if (!(d > 0.0f)) return false;
        A[static_cast<std::size_t>(i) * L + j] = cplx(std::sqrt(d), 0.0f);
      } else {
        A[static_cast<std::size_t>(i) * L + j] =
            sum / A[static_cast<std::size_t>(j) * L + j];
      }
    }
  }
  // Forward substitution: L y = b
  for (int i = 0; i < L; ++i) {
    cplx sum = b[static_cast<std::size_t>(i)];
    for (int k = 0; k < i; ++k)
      sum -= A[static_cast<std::size_t>(i) * L + k] * b[static_cast<std::size_t>(k)];
    b[static_cast<std::size_t>(i)] = sum / A[static_cast<std::size_t>(i) * L + i];
  }
  // Back substitution: L^H x = y
  for (int i = L - 1; i >= 0; --i) {
    cplx sum = b[static_cast<std::size_t>(i)];
    for (int k = i + 1; k < L; ++k)
      sum -= std::conj(A[static_cast<std::size_t>(k) * L + i]) * b[static_cast<std::size_t>(k)];
    b[static_cast<std::size_t>(i)] = sum / A[static_cast<std::size_t>(i) * L + i];
  }
  return true;
}

// --- Channel combiners ------------------------------------------------------

cplx combine_das(const cplx* y, int n) {
  cplx acc(0, 0);
  for (int i = 0; i < n; ++i) acc += y[i];
  return acc;
}

cplx combine_dmas(const cplx* y, int n, int n_valid) {
  // Delay-multiply-and-sum: sum over unordered channel pairs of the pair product
  // normalised back to signal units,
  //
  //     yhat_ij = y_i * y_j / sqrt(|y_i| * |y_j|),
  //
  // which on real RF is exactly the textbook sign(y_i*y_j)*sqrt(|y_i*y_j|). The
  // square root restores the units the multiplication squared; it does *not*
  // halve the phase. Getting that wrong (by taking the principal root of the
  // product instead) halves the phase ramp across the aperture and makes the
  // mainlobe wider than DAS rather than narrower — the opposite of the point.
  //
  // Substituting v_i = y_i / sqrt(|y_i|) makes yhat_ij = v_i * v_j, so
  //
  //     sum_{i<j} v_i v_j = ( (sum_i v_i)^2 - sum_i v_i^2 ) / 2,
  //
  // which evaluates in one pass instead of the naive O(N^2) double loop. The
  // squared coherent sum is where the contrast and resolution gain comes from:
  // the aperture sum is squared, so its mainlobe narrows and its sidelobes fall
  // off twice as fast in dB.
  //
  // One consequence to be aware of downstream: the pair product doubles the
  // residual phase, so the output oscillates axially at twice the rate of the
  // DAS image. On RF this is why the technique is always followed by a bandpass
  // at 2*f0; on baseband IQ the carrier is already removed, so DC stays at DC
  // and no extra filter is needed — but the frame is no longer phase-linear in
  // displacement, which is why the quantitative estimators in usx/doppler.hpp
  // should be run on DAS frames, not DMAS ones.
  cplx s(0, 0), q(0, 0);
  for (int i = 0; i < n; ++i) {
    const real m = std::abs(y[i]);
    if (m <= 0.0f) continue;
    const cplx v = y[i] / std::sqrt(m);
    s += v;
    q += v * v;
  }
  const cplx acc = (s * s - q) * 0.5f;
  // DMAS grows like N^2 where DAS grows like N; rescale so the two are
  // comparable in absolute magnitude and can share a display dynamic range.
  const real norm = 0.5f * std::max(1.0f, static_cast<real>(n_valid - 1));
  return acc / norm;
}

// Minimum variance over the *active* channel subset.
//
// `Y` is the block scratch buffer, laid out as Y[ch * W + k] where k indexes the
// depth window; `kw` is the pixel being combined. Reading the neighbouring
// depths directly out of that buffer is what makes range averaging free.
cplx combine_mv_windowed(const cplx* Y, int W, int kw, int n_ch, const int* active, int n_active,
                         int subarray, real loading, int range_avg, int w_n,
                         std::vector<cplx>& R, std::vector<cplx>& w, std::vector<cplx>& sub) {
  auto gather = [&](int k, int ch) { return Y[static_cast<std::size_t>(ch) * W + k]; };

  if (n_active < 4) {
    cplx acc(0, 0);
    for (int a = 0; a < n_active; ++a) acc += gather(kw, active[a]);
    return acc;
  }
  const int L = std::clamp(subarray, 2, std::max(2, n_active / 2));
  const int n_sub = n_active - L + 1;

  R.assign(static_cast<std::size_t>(L) * L, cplx(0, 0));
  sub.assign(static_cast<std::size_t>(L), cplx(0, 0));

  // Spatial smoothing: average the outer product over every overlapping
  // subarray of the aperture, and over neighbouring range samples. Without this
  // the estimate is rank 1 and cannot be inverted; with it, coherent
  // interference is also decorrelated, which is the other reason the technique
  // is required rather than merely helpful.
  int n_snap = 0;
  for (int d = -range_avg; d <= range_avg; ++d) {
    const int k = kw + d;
    if (k < 0 || k >= w_n) continue;
    for (int s0 = 0; s0 < n_sub; ++s0) {
      for (int a = 0; a < L; ++a)
        sub[static_cast<std::size_t>(a)] = gather(k, active[s0 + a]);
      for (int a = 0; a < L; ++a) {
        const cplx va = sub[static_cast<std::size_t>(a)];
        for (int b = a; b < L; ++b)
          R[static_cast<std::size_t>(a) * L + b] +=
              va * std::conj(sub[static_cast<std::size_t>(b)]);
      }
      ++n_snap;
    }
  }
  if (n_snap == 0) return cplx(0, 0);

  const real inv = 1.0f / static_cast<real>(n_snap);
  real trace = 0.0f;
  for (int a = 0; a < L; ++a) {
    for (int b = a; b < L; ++b) {
      R[static_cast<std::size_t>(a) * L + b] *= inv;
      if (b != a)
        R[static_cast<std::size_t>(b) * L + a] =
            std::conj(R[static_cast<std::size_t>(a) * L + b]);
    }
    trace += R[static_cast<std::size_t>(a) * L + a].real();
  }
  const real eps = std::max(loading * trace / static_cast<real>(L), 1e-20f);
  for (int a = 0; a < L; ++a) R[static_cast<std::size_t>(a) * L + a] += cplx(eps, 0.0f);

  // The data are already delay-compensated, so the look direction is the
  // all-ones steering vector and the constrained optimum is
  //     w = R^-1 a / (a^H R^-1 a),   a = [1 ... 1]^T.
  w.assign(static_cast<std::size_t>(L), cplx(1, 0));
  auto das_fallback = [&] {
    cplx acc(0, 0);
    for (int a = 0; a < n_active; ++a) acc += gather(kw, active[a]);
    return acc;
  };
  if (!cholesky_solve(R, L, w)) return das_fallback();
  cplx denom(0, 0);
  for (int a = 0; a < L; ++a) denom += w[static_cast<std::size_t>(a)];
  if (std::abs(denom) < 1e-30f) return das_fallback();

  // Apply the weights to the subarray-averaged snapshot.
  cplx out(0, 0);
  for (int a = 0; a < L; ++a) {
    cplx avg(0, 0);
    for (int s0 = 0; s0 < n_sub; ++s0) avg += gather(kw, active[s0 + a]);
    avg /= static_cast<real>(n_sub);
    out += std::conj(w[static_cast<std::size_t>(a)]) * avg;
  }
  // Rescale to DAS-like magnitude so the combiners share a display range.
  return out / denom * static_cast<real>(n_active);
}

}  // namespace

void BeamformerConfig::validate() const {
  if (f_number <= 0) throw std::invalid_argument("BeamformerConfig.f_number must be > 0");
  if (mv_subarray < 2) throw std::invalid_argument("BeamformerConfig.mv_subarray must be >= 2");
  if (mv_diagonal_loading < 0)
    throw std::invalid_argument("BeamformerConfig.mv_diagonal_loading must be >= 0");
  if (mv_range_average < 0)
    throw std::invalid_argument("BeamformerConfig.mv_range_average must be >= 0");
}

namespace {

// Lateral extent of a transmit's firing aperture, computed once per event
// instead of once per pixel. Before this was hoisted it was the single largest
// cost in the beamformer: an O(n_elements) scan inside a loop that already runs
// once per pixel per transmit.
struct TxExtent {
  real x_lo = 0, x_hi = 0;
  bool valid = false;
};

TxExtent compute_tx_extent(const Probe& probe, const Transmit& tx) {
  TxExtent e;
  for (int i = 0; i < probe.n_elements; ++i) {
    if (!tx.apodization.empty() && tx.apodization[static_cast<std::size_t>(i)] == 0.0f) continue;
    const real x = probe.element_position(i).x;
    if (!e.valid) { e.x_lo = e.x_hi = x; e.valid = true; }
    else { e.x_lo = std::min(e.x_lo, x); e.x_hi = std::max(e.x_hi, x); }
  }
  return e;
}

real tx_sensitivity(const TxExtent& e, const Transmit& tx, const Vec3& point, real wavelength,
                    real pitch) {
  if (!e.valid || point.z <= 0.0f) return 0.0f;

  real a, b;
  if (tx.type == TransmitType::PlaneWave) {
    // A plane wave illuminates the aperture footprint translated along the
    // steering direction. Outside that parallelogram there is simply no
    // transmitted energy, and any signal a beamformer finds there is clutter.
    const real shift = point.z * std::tan(tx.angle);
    a = e.x_lo + shift;
    b = e.x_hi + shift;
  } else {
    // Focused or diverging: trace the rays from the virtual source through the
    // aperture edges down to this depth.
    const real zf = tx.focus.z;
    if (std::abs(zf) < 1e-9f) return 1.0f;
    a = e.x_lo + (tx.focus.x - e.x_lo) * point.z / zf;
    b = e.x_hi + (tx.focus.x - e.x_hi) * point.z / zf;
    if (a > b) std::swap(a, b);
  }

  // Widen by the diffraction-limited beam width, otherwise a focused beam has
  // exactly zero width at its focus and the pixels there would be discarded.
  // The width comes from the *aperture* that formed the beam (f-number times
  // wavelength), not from the geometric ray spread at this depth — using the
  // latter blows up to metres at the focus, where the ray spread is zero.
  const real aperture = std::max(e.x_hi - e.x_lo, pitch);
  const real f_num_eff = std::clamp(point.z / aperture, 0.5f, 20.0f);
  const real diffraction = f_num_eff * wavelength;
  a -= diffraction;
  b += diffraction;

  const real centre = 0.5f * (a + b);
  const real half = 0.5f * (b - a);
  if (half <= 0.0f) return 0.0f;
  // Tukey taper: flat across the illuminated region, smoothly to zero at the
  // edges, so that neighbouring transmits blend instead of leaving seams.
  return window_value(Window::Tukey, (point.x - centre) / half, 0.4f);
}

}  // namespace

real transmit_sensitivity(const Probe& probe, const Transmit& tx, const Vec3& point,
                          real wavelength) {
  return tx_sensitivity(compute_tx_extent(probe, tx), tx, point, wavelength, probe.pitch);
}

Frame Beamformer::beamform(const ChannelData& data, const ScanGrid& grid,
                           int event_begin, int event_end) const {
  data.validate();
  grid.validate();
  const int n_ev_total = data.n_events();
  if (event_end < 0) event_end = n_ev_total;
  event_begin = std::clamp(event_begin, 0, n_ev_total);
  event_end = std::clamp(event_end, event_begin, n_ev_total);

  const int n_ch = data.n_channels();
  const int n_smp = data.n_samples();
  const real c = data.medium.speed_of_sound;
  const real fs = data.acq.sampling_frequency;
  const real t0 = data.acq.t0;
  const real fdem = data.is_iq ? data.demodulation_frequency : 0.0f;
  const real lambda = data.probe.wavelength(c);
  const real pitch = data.probe.pitch;

  std::vector<Vec3> elem(static_cast<std::size_t>(n_ch));
  for (int i = 0; i < n_ch; ++i) elem[static_cast<std::size_t>(i)] = data.probe.element_position(i);

  std::vector<TxExtent> extents(static_cast<std::size_t>(n_ev_total));
  for (int ev = event_begin; ev < event_end; ++ev)
    extents[static_cast<std::size_t>(ev)] = compute_tx_extent(data.probe, data.transmits[ev]);

  Frame frame;
  frame.grid = grid;
  frame.demodulation_frequency = fdem;
  frame.allocate();

  const int n1 = grid.n_axis1;
  const bool incoherent = config_.compounding == Compounding::Incoherent;
  const bool select = config_.compounding == Compounding::Select;
  const bool is_mv = config_.combiner == Combiner::MV;
  const int range_avg = is_mv ? config_.mv_range_average : 0;

  // The line is processed in depth blocks rather than all at once. Within a
  // block the inner loop walks one channel's trace in increasing depth, which is
  // very nearly a sequential read of the raw data, and writes into a scratch
  // buffer small enough to stay resident in cache. Looping the other way round
  // (pixel outer, channel inner) touches n_channels widely separated cache lines
  // per pixel and was measured to be several times slower on the same work.
  constexpr int kBlock = 64;
  const int pad = range_avg;  // halo so covariance averaging can cross block edges
  const int W = kBlock + 2 * pad;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    const std::size_t buf = static_cast<std::size_t>(n_ch) * W;
    std::vector<cplx> Y(buf);
    std::vector<real> apod(buf), rx_tau(buf), rx_cos(buf), rx_sin(buf);
    std::vector<real> tx_tau(static_cast<std::size_t>(W)), tx_w(static_cast<std::size_t>(W));
    std::vector<real> tx_cos(static_cast<std::size_t>(W)), tx_sin(static_cast<std::size_t>(W));
    std::vector<int> best_ev(static_cast<std::size_t>(W));
    std::vector<real> best_w(static_cast<std::size_t>(W));
    std::vector<cplx> ybuf(static_cast<std::size_t>(n_ch));
    std::vector<int> active(static_cast<std::size_t>(n_ch));
    std::vector<cplx> R, w, sub;
    std::vector<const cplx*> extra(static_cast<std::size_t>(2 * range_avg + 1), nullptr);
    std::vector<real> block_acc(static_cast<std::size_t>(kBlock));

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
    for (int i0 = 0; i0 < grid.n_axis0; ++i0) {
      const Vec3* line = grid.points.data() + static_cast<std::size_t>(i0) * n1;
      cplx* out_line = frame.data.data() + static_cast<std::size_t>(i0) * n1;

      for (int j0 = 0; j0 < n1; j0 += kBlock) {
        const int core_n = std::min(kBlock, n1 - j0);
        const int w_lo = std::max(0, j0 - pad);
        const int w_hi = std::min(n1, j0 + core_n + pad);
        const int w_n = w_hi - w_lo;
        const int core_off = j0 - w_lo;  // index of the first output pixel within the window

        // ---- Receive geometry: depends on the pixel and channel only, so it
        // is computed once and reused across every transmit event. -----------
        for (int ch = 0; ch < n_ch; ++ch) {
          const Vec3& pe = elem[static_cast<std::size_t>(ch)];
          const std::size_t base = static_cast<std::size_t>(ch) * W;
          for (int k = 0; k < w_n; ++k) {
            const Vec3& p = line[w_lo + k];
            const real a = dynamic_aperture_weight(pe, p, config_.f_number, config_.rx_window,
                                                   config_.tukey_alpha);
            // Minimum variance derives its own weights, so it is given a plain
            // rectangular aperture; pre-tapering would fight the optimiser.
            apod[base + k] = is_mv ? (a > 0.0f ? 1.0f : 0.0f) : a;
            if (a <= 0.0f) continue;
            const real tau = distance(pe, p) / c;
            rx_tau[base + k] = tau;
            if (fdem > 0.0f)
              fast_sincos(2.0f * kPi * fdem * tau, &rx_sin[base + k], &rx_cos[base + k]);
          }
        }

        if (select) {
          std::fill(best_ev.begin(), best_ev.begin() + w_n, -1);
          std::fill(best_w.begin(), best_w.begin() + w_n, 0.0f);
          for (int ev = event_begin; ev < event_end; ++ev) {
            const Transmit& tx = data.transmits[static_cast<std::size_t>(ev)];
            const TxExtent& ex = extents[static_cast<std::size_t>(ev)];
            for (int k = 0; k < w_n; ++k) {
              const real tw = tx_sensitivity(ex, tx, line[w_lo + k], lambda, pitch);
              if (tw > best_w[static_cast<std::size_t>(k)]) {
                best_w[static_cast<std::size_t>(k)] = tw;
                best_ev[static_cast<std::size_t>(k)] = ev;
              }
            }
          }
        }

        const int n_passes = incoherent ? (event_end - event_begin) : 1;
        std::fill(block_acc.begin(), block_acc.end(), 0.0f);

        for (int pass = 0; pass < std::max(1, n_passes); ++pass) {
          std::fill(Y.begin(), Y.end(), cplx(0, 0));

          const int ev_lo = incoherent ? event_begin + pass : event_begin;
          const int ev_hi = incoherent ? ev_lo + 1 : event_end;

          for (int ev = ev_lo; ev < ev_hi; ++ev) {
            const Transmit& tx = data.transmits[static_cast<std::size_t>(ev)];
            const TxExtent& ex = extents[static_cast<std::size_t>(ev)];

            // ---- Transmit geometry: depends on the pixel and event only. ----
            bool any = false;
            for (int k = 0; k < w_n; ++k) {
              const Vec3& p = line[w_lo + k];
              real tw;
              if (select) {
                tw = (best_ev[static_cast<std::size_t>(k)] == ev)
                         ? best_w[static_cast<std::size_t>(k)]
                         : 0.0f;
                // In Select mode the chosen beam is used at full weight; the
                // taper's only job there was to pick the winner.
                if (tw > 0.0f) tw = 1.0f;
              } else if (config_.transmit_weighting) {
                tw = tx_sensitivity(ex, tx, p, lambda, pitch);
              } else {
                tw = 1.0f;
              }
              tx_w[static_cast<std::size_t>(k)] = tw;
              if (tw <= 0.0f) continue;
              any = true;
              const real tau = transmit_arrival_time(tx, p, c);
              tx_tau[static_cast<std::size_t>(k)] = tau;
              if (fdem > 0.0f)
                fast_sincos(2.0f * kPi * fdem * tau, &tx_sin[static_cast<std::size_t>(k)],
                            &tx_cos[static_cast<std::size_t>(k)]);
              else {
                tx_cos[static_cast<std::size_t>(k)] = 1.0f;
                tx_sin[static_cast<std::size_t>(k)] = 0.0f;
              }
            }
            if (!any) continue;

            // ---- The hot loop: one channel's trace, walked in depth. --------
            for (int ch = 0; ch < n_ch; ++ch) {
              const std::size_t base = static_cast<std::size_t>(ch) * W;
              const real* ap = apod.data() + base;
              const real* rt = rx_tau.data() + base;
              const real* rc = rx_cos.data() + base;
              const real* rs = rx_sin.data() + base;
              cplx* yy = Y.data() + base;

              if (data.is_iq) {
                const cplx* trace = data.iq.data() + data.index(ev, ch, 0);
                for (int k = 0; k < w_n; ++k) {
                  const real a = ap[k];
                  const real tw = tx_w[static_cast<std::size_t>(k)];
                  if (a <= 0.0f || tw <= 0.0f) continue;
                  const real idx = (tx_tau[static_cast<std::size_t>(k)] + rt[k] - t0) * fs;
                  cplx v = interp(trace, n_smp, idx);
                  // Undo the phase the mixer imposed on this delay. Without this
                  // the channels are aligned in envelope but not in phase, and
                  // they cancel instead of summing — the classic symptom is an
                  // image that gets darker as the aperture grows.
                  const real tc = tx_cos[static_cast<std::size_t>(k)];
                  const real ts = tx_sin[static_cast<std::size_t>(k)];
                  yy[k] += v * cplx(tc * rc[k] - ts * rs[k], ts * rc[k] + tc * rs[k]) * (a * tw);
                }
              } else {
                const real* trace = data.rf.data() + data.index(ev, ch, 0);
                for (int k = 0; k < w_n; ++k) {
                  const real a = ap[k];
                  const real tw = tx_w[static_cast<std::size_t>(k)];
                  if (a <= 0.0f || tw <= 0.0f) continue;
                  const real idx = (tx_tau[static_cast<std::size_t>(k)] + rt[k] - t0) * fs;
                  yy[k] += interp_rf(trace, n_smp, idx) * (a * tw);
                }
              }
            }
          }

          // ---- Combine the aligned channels into a pixel value -------------
          for (int k = 0; k < core_n; ++k) {
            const int kw = core_off + k;
            int n_active = 0;
            for (int ch = 0; ch < n_ch; ++ch) {
              if (apod[static_cast<std::size_t>(ch) * W + kw] <= 0.0f) continue;
              active[static_cast<std::size_t>(n_active++)] = ch;
              ybuf[static_cast<std::size_t>(ch)] = Y[static_cast<std::size_t>(ch) * W + kw];
            }
            if (n_active == 0) {
              if (!incoherent) out_line[j0 + k] = cplx(0, 0);
              continue;
            }
            for (int ch = 0; ch < n_ch; ++ch)
              if (apod[static_cast<std::size_t>(ch) * W + kw] <= 0.0f)
                ybuf[static_cast<std::size_t>(ch)] = cplx(0, 0);

            cplx out(0, 0);
            switch (config_.combiner) {
              case Combiner::DAS:
                out = combine_das(ybuf.data(), n_ch);
                break;
              case Combiner::DMAS:
                out = combine_dmas(ybuf.data(), n_ch, n_active);
                break;
              case Combiner::MV:
                out = combine_mv_windowed(Y.data(), W, kw, n_ch, active.data(), n_active,
                                          config_.mv_subarray, config_.mv_diagonal_loading,
                                          range_avg, w_n, R, w, sub);
                break;
            }

            if (config_.coherence_factor) {
              cplx coh(0, 0);
              real inc = 0.0f;
              for (int a = 0; a < n_active; ++a) {
                const cplx& v = ybuf[static_cast<std::size_t>(active[static_cast<std::size_t>(a)])];
                coh += v;
                inc += std::norm(v);
              }
              out *= (inc > 0.0f) ? std::norm(coh) / (static_cast<real>(n_active) * inc) : 0.0f;
            }

            if (incoherent) block_acc[static_cast<std::size_t>(k)] += std::norm(out);
            else out_line[j0 + k] = out;
          }
        }

        if (incoherent)
          for (int k = 0; k < core_n; ++k)
            out_line[j0 + k] = cplx(std::sqrt(block_acc[static_cast<std::size_t>(k)]), 0.0f);
      }
    }
  }

  return frame;
}

std::vector<Frame> Beamformer::beamform_sequence(const ChannelData& data, const ScanGrid& grid,
                                                 int events_per_frame) const {
  if (events_per_frame <= 0)
    throw std::invalid_argument("beamform_sequence: events_per_frame must be > 0");
  const int n_ev = data.n_events();
  const int n_frames = n_ev / events_per_frame;
  if (n_frames == 0)
    throw std::invalid_argument("beamform_sequence: fewer events than events_per_frame");

  std::vector<Frame> frames;
  frames.reserve(static_cast<std::size_t>(n_frames));
  for (int f = 0; f < n_frames; ++f) {
    Frame fr = beamform(data, grid, f * events_per_frame, (f + 1) * events_per_frame);
    fr.sequence = f;
    fr.timestamp = static_cast<double>(f) * events_per_frame / data.acq.prf;
    frames.push_back(std::move(fr));
  }
  return frames;
}

}  // namespace usx
