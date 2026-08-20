// usx/beamformer.hpp — Delay, apodize, and combine.
//
// Beamforming is the step that turns per-element echo traces into an image. For
// each pixel it asks: if there were a scatterer *here*, when would its echo
// have reached each element? It then reads each channel at that time, so that
// the wanted echo lines up across all channels while everything else does not,
// and combines the aligned channels into one value.
//
// The delay part is fixed by geometry. The *combining* part is where the design
// space is, and this header exposes three points in it:
//
//   DAS  — sum the aligned channels with a fixed apodization window. Linear,
//          data-independent, trivially parallel, and what essentially every
//          clinical scanner ships. Its resolution and sidelobe level are set
//          entirely by the aperture and the window.
//
//   DMAS — multiply every pair of channels, take the signed square root of each
//          product, and sum. Pairs that are genuinely coherent reinforce; a pair
//          whose alignment is accidental does not. It is effectively squaring
//          the aperture sum, so the mainlobe narrows and the sidelobes fall off
//          twice as fast in dB. The price is a non-linear response that is
//          harder to characterise and no longer phase-linear in displacement.
//
//   MV   — minimum variance (Capon). Estimate the spatial covariance of the
//          aligned channels and choose the weights that minimise total output
//          power subject to unity gain on the look direction. Adaptive: it
//          places nulls on whatever is interfering. Dramatically sharper, but
//          data-dependent, expensive, and it will happily null out the signal
//          itself if the covariance estimate is poor — hence subarray averaging
//          and diagonal loading, which are not optional extras but the price of
//          making it work at all.
//
// The engine's output is complex, never a picture. See usx/types.hpp, Frame.
#pragma once

#include <vector>

#include "usx/types.hpp"

namespace usx {

enum class Combiner : std::uint8_t {
  DAS = 0,   // delay and sum
  DMAS = 1,  // delay, multiply and sum (filtered-delay-multiply-and-sum)
  MV = 2,    // minimum variance / Capon
};

// How to merge multiple transmit events into one image.
enum class Compounding : std::uint8_t {
  // Sum the aligned channel data across transmits before combining. Correct for
  // steered plane waves: the transmits differ only in wavefront angle, so their
  // echoes from the same scatterer are phase-coherent and add up like a
  // synthetic transmit focus. This is what makes plane-wave compounding recover
  // focused-quality images at ultrafast frame rates.
  Coherent = 0,
  // Beamform each transmit separately and sum the magnitudes. Destroys phase
  // (so no Doppler downstream) but is robust to transmits that are not mutually
  // coherent, and reduces speckle variance.
  Incoherent = 1,
  // Use only the single best-illuminating transmit per pixel. This is classic
  // line-by-line focused imaging expressed on an arbitrary grid: no blending
  // artefacts between beams, phase preserved.
  Select = 2,
};

struct BeamformerConfig {
  // Receive f-number: aperture opens with depth to hold this constant. Lower
  // means a wider aperture and finer lateral resolution, until the outermost
  // channels are so oblique that element directivity and phase error make them
  // do more harm than good.
  real f_number = 1.75f;
  Window rx_window = Window::Hann;
  real tukey_alpha = 0.5f;

  Combiner combiner = Combiner::DAS;
  Compounding compounding = Compounding::Coherent;

  // Weight each transmit's contribution by how well it illuminates the pixel.
  // Without this, a plane wave contributes noise to pixels outside the region
  // it actually insonified, and adjacent focused beams bleed into each other.
  bool transmit_weighting = true;

  // Multiply the output by the coherence factor,
  //     CF = |sum_i y_i|^2 / (N * sum_i |y_i|^2),
  // the ratio of coherent to incoherent energy across the aperture. It is near
  // 1 where the channels agree (a real target) and near 0 where they do not
  // (clutter, off-axis energy). Cheap, effective, and aggressive — it darkens
  // speckle as well as clutter, so it flatters contrast measurements while
  // changing the texture statistics the eye relies on.
  bool coherence_factor = false;

  // --- Minimum variance parameters -----------------------------------------
  // Subarray (spatial smoothing) length. The covariance of a single snapshot is
  // rank 1 and cannot be inverted, so it is estimated by averaging over
  // overlapping subarrays of the aperture. Shorter subarrays give a better
  // conditioned estimate and worse resolution; L <= N/2 is the usual rule.
  int mv_subarray = 16;
  // Diagonal loading as a fraction of trace(R)/L. This is what stops the
  // adaptive weights from nulling the signal when the covariance estimate is
  // noisy; it continuously trades MV back towards DAS as it increases.
  real mv_diagonal_loading = 1e-2f;
  // Additional averaging of the covariance over +/- this many range samples.
  int mv_range_average = 2;

  void validate() const;
};

class Beamformer {
 public:
  explicit Beamformer(BeamformerConfig config = {}) : config_(config) { config_.validate(); }

  // Beamform transmit events [event_begin, event_end) into one frame.
  // event_end < 0 means "to the end".
  Frame beamform(const ChannelData& data, const ScanGrid& grid,
                 int event_begin = 0, int event_end = -1) const;

  // Split the acquisition into consecutive groups of `events_per_frame`
  // transmits and beamform each into its own frame. This is how a Doppler
  // ensemble is formed: N repetitions of the same (possibly compounded)
  // transmit sequence become N complex frames whose pixel-wise phase evolution
  // encodes motion.
  std::vector<Frame> beamform_sequence(const ChannelData& data, const ScanGrid& grid,
                                       int events_per_frame) const;

  const BeamformerConfig& config() const { return config_; }
  BeamformerConfig& config() { return config_; }

 private:
  BeamformerConfig config_;
};

// Illumination weight of a transmit at a point: 1 inside the insonified region,
// tapering to 0 outside it. Exposed because it is also what a caller needs to
// build a transmit-coverage mask or to normalise a compounded image.
real transmit_sensitivity(const Probe& probe, const Transmit& tx, const Vec3& point,
                          real wavelength);

}  // namespace usx
