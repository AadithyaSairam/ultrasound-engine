// usx/demodulate.hpp — RF to baseband IQ conversion.
//
// Raw RF is sampled at several times the transducer centre frequency, but the
// *information* in it occupies only the transducer's bandwidth around f0.
// Mixing that band down to DC and low-pass filtering it gives complex baseband
// IQ: the same information at a fraction of the sample rate.
//
// This matters far more than a compression trick. A beamformer has to resample
// each channel at an arbitrary, continuously varying delay. On RF, interpolating
// between samples of a 5 MHz carrier introduces phase error unless you either
// oversample heavily or use an expensive interpolator. On IQ the carrier has
// been removed, the remaining envelope is slowly varying, cheap linear
// interpolation is accurate, and the sub-sample delay becomes an exact phase
// rotation. Every real-time ultrasound system does its beamforming this way.
#pragma once

#include <vector>

#include "usx/types.hpp"

namespace usx {

struct DemodConfig {
  // Frequency to mix down from. Defaults to the probe centre frequency when 0.
  real demodulation_frequency = 0.0f;

  // Low-pass cutoff as a fraction of the demodulation frequency. The baseband
  // signal occupies +/- B/2 where B is the pulse bandwidth, so the cutoff must
  // clear that but stay below f0 to reject the image at -2*f0.
  real cutoff_fraction = 0.5f;

  // FIR length. Longer means a sharper transition and more group delay (which
  // is compensated, being linear phase).
  int filter_taps = 33;

  // Keep every Nth sample. The Nyquist limit after filtering is the baseband
  // bandwidth, so decimation of 2-4 is typically free. Decimating too hard
  // aliases the pulse and smears axial resolution.
  int decimation = 1;
};

// Convert real RF channel data to complex baseband IQ. The returned
// ChannelData carries an updated sampling_frequency and n_samples if decimated,
// and records the demodulation frequency the beamformer needs.
ChannelData demodulate(const ChannelData& rf, DemodConfig config = {});

// Design a windowed-sinc low-pass FIR. Exposed for testing and for the Doppler
// wall filter, which uses the same machinery.
std::vector<real> design_lowpass(int n_taps, real normalized_cutoff);

}  // namespace usx
