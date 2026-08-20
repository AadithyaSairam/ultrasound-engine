#include "usx/demodulate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace usx {

std::vector<real> design_lowpass(int n_taps, real normalized_cutoff) {
  if (n_taps < 3) n_taps = 3;
  if (n_taps % 2 == 0) ++n_taps;  // odd length keeps the delay an integer
  const real fc = std::clamp(normalized_cutoff, 1e-4f, 0.499f);
  std::vector<real> h(static_cast<std::size_t>(n_taps));
  const int m = n_taps / 2;
  real sum = 0.0f;
  for (int i = 0; i < n_taps; ++i) {
    const int k = i - m;
    // Ideal low-pass impulse response (a sinc) truncated by a Hamming window.
    // Truncating without a window would give large Gibbs ripple in the stopband,
    // which shows up as leakage of the -2*f0 image back into the IQ.
    const real sinc = (k == 0) ? 2.0f * fc
                               : std::sin(2.0f * kPi * fc * k) / (kPi * k);
    const real w = 0.54f - 0.46f * std::cos(2.0f * kPi * i / (n_taps - 1));
    h[static_cast<std::size_t>(i)] = sinc * w;
    sum += sinc * w;
  }
  if (sum != 0.0f)
    for (real& v : h) v /= sum;  // unity DC gain
  return h;
}

ChannelData demodulate(const ChannelData& rf, DemodConfig config) {
  rf.validate();
  if (rf.is_iq) return rf;
  if (rf.rf.empty()) throw std::invalid_argument("demodulate: input has no RF samples");

  const real fdem = config.demodulation_frequency > 0.0f
                        ? config.demodulation_frequency
                        : rf.probe.center_frequency;
  const int dec = std::max(1, config.decimation);
  const real fs = rf.acq.sampling_frequency;

  const std::vector<real> h = design_lowpass(config.filter_taps,
                                             config.cutoff_fraction * fdem / fs);
  const int n_taps = static_cast<int>(h.size());
  const int group_delay = n_taps / 2;

  ChannelData out;
  out.probe = rf.probe;
  out.medium = rf.medium;
  out.transmits = rf.transmits;
  out.acq = rf.acq;
  out.acq.sampling_frequency = fs / dec;
  out.acq.n_samples = (rf.acq.n_samples + dec - 1) / dec;
  out.is_iq = true;
  out.demodulation_frequency = fdem;
  out.allocate();

  const int n_ev = rf.n_events(), n_ch = rf.n_channels(), n_in = rf.n_samples();
  const int n_out = out.acq.n_samples;

  // The mixing phase depends only on the sample index, never on which channel or
  // event the sample came from, so the rotator is tabulated once and reused
  // across the whole acquisition. Recomputing sin/cos per sample instead costs
  // n_events * n_channels * n_samples transcendental calls — for a modest
  // 6 x 128 x 3072 acquisition that is 2.4 million of them, and it dominated the
  // stage before this was hoisted.
  std::vector<cplx> rotator(static_cast<std::size_t>(n_in));
  for (int n = 0; n < n_in; ++n) {
    const real ph = -2.0f * kPi * fdem * rf.acq.sample_time(n);
    rotator[static_cast<std::size_t>(n)] = cplx(std::cos(ph), std::sin(ph));
  }

  const std::size_t n_traces = static_cast<std::size_t>(n_ev) * n_ch;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<cplx> mixed(static_cast<std::size_t>(n_in));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t trace = 0; trace < static_cast<std::ptrdiff_t>(n_traces); ++trace) {
      const int ev = static_cast<int>(trace / n_ch);
      const int ch = static_cast<int>(trace % n_ch);
      const real* src = rf.rf.data() + rf.index(ev, ch, 0);

      // Mix: multiplying by exp(-j*2*pi*f*t) shifts the +f0 band to DC and the
      // -f0 band to -2*f0, where the low-pass then removes it. The absolute time
      // (not the sample index) is used so that the phase reference is the
      // acquisition clock, which is what makes the beamformer's phase rotation
      // consistent across events with different t0.
      for (int n = 0; n < n_in; ++n)
        mixed[static_cast<std::size_t>(n)] = rotator[static_cast<std::size_t>(n)] * src[n];

      // Filter and decimate in one pass, compensating the linear-phase group
      // delay so that IQ sample k still corresponds to acquisition time
      // t0 + k*dec/fs. Skipping this compensation shifts every echo axially by
      // half the filter length, which reads as a depth calibration error.
      cplx* dst = out.iq.data() + out.index(ev, ch, 0);
      for (int k = 0; k < n_out; ++k) {
        const int centre = k * dec + group_delay;
        const int j_lo = std::max(0, centre - n_in + 1);
        const int j_hi = std::min(n_taps - 1, centre);
        cplx acc(0, 0);
        for (int j = j_lo; j <= j_hi; ++j)
          acc += mixed[static_cast<std::size_t>(centre - j)] * h[static_cast<std::size_t>(j)];
        // Factor of 2 restores the original envelope amplitude: the real
        // passband carries half its energy at -f0, which the filter discarded.
        dst[k] = acc * 2.0f;
      }
    }
  }

  return out;
}

}  // namespace usx
