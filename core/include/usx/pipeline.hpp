// usx/pipeline.hpp — The streaming front end.
//
// Everything else in the engine is a stateless transform. This is the piece
// that holds the configuration, owns the scratch buffers, and turns a stream of
// raw acquisitions into a stream of frames at a measurable frame rate. It is
// what a live application talks to.
//
// It keeps two things a stateless function cannot:
//   - reusable buffers, so a steady-state frame allocates nothing;
//   - a smoothed brightness reference, so the display does not flicker when a
//     bright specular reflector enters or leaves the field of view.
#pragma once

#include <chrono>
#include <vector>

#include "usx/beamformer.hpp"
#include "usx/demodulate.hpp"
#include "usx/doppler.hpp"
#include "usx/postprocess.hpp"
#include "usx/types.hpp"

namespace usx {

struct PipelineConfig {
  DemodConfig demod;
  BeamformerConfig beamformer;

  real dynamic_range_db = 60.0f;
  real gain_db = 0.0f;
  bool apply_tgc = true;

  // Smooth the display reference level across frames. 0 re-normalises every
  // frame independently (which makes the brightness jump); 1 freezes it.
  real reference_smoothing = 0.85f;
};

struct PipelineStats {
  double demodulate_ms = 0;
  double beamform_ms = 0;
  double postprocess_ms = 0;
  double total_ms = 0;
  double frames_per_second() const { return total_ms > 0 ? 1000.0 / total_ms : 0.0; }
};

class ImagingPipeline {
 public:
  ImagingPipeline(ScanGrid grid, PipelineConfig config = {});

  // Raw channel data (RF or already-IQ) to a complex beamformed frame. This is
  // the output that downstream quantitative work consumes.
  Frame process_frame(const ChannelData& data);

  // Raw channel data all the way to a log-compressed display image in [0, 1].
  std::vector<real> process_display(const ChannelData& data);

  // Convert an already-beamformed frame to a display image, reusing the
  // pipeline's smoothed reference level.
  std::vector<real> to_display(const Frame& frame, const Medium& medium);

  const ScanGrid& grid() const { return grid_; }
  const PipelineStats& stats() const { return stats_; }
  PipelineConfig& config() { return config_; }
  const PipelineConfig& config() const { return config_; }
  std::int64_t frame_count() const { return frame_count_; }
  void reset();

 private:
  ScanGrid grid_;
  PipelineConfig config_;
  Beamformer beamformer_;
  PipelineStats stats_;
  real reference_ = 0.0f;
  std::int64_t frame_count_ = 0;
};

}  // namespace usx
