#include "usx/pipeline.hpp"

#include <algorithm>
#include <cmath>

namespace usx {
namespace {
using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
}  // namespace

ImagingPipeline::ImagingPipeline(ScanGrid grid, PipelineConfig config)
    : grid_(std::move(grid)), config_(config), beamformer_(config.beamformer) {
  grid_.validate();
}

void ImagingPipeline::reset() {
  reference_ = 0.0f;
  frame_count_ = 0;
  stats_ = PipelineStats{};
}

Frame ImagingPipeline::process_frame(const ChannelData& data) {
  const auto t_start = Clock::time_point(Clock::now());
  stats_ = PipelineStats{};

  const ChannelData* iq_ptr = &data;
  ChannelData iq_storage;
  if (!data.is_iq) {
    const auto t = Clock::now();
    iq_storage = demodulate(data, config_.demod);
    iq_ptr = &iq_storage;
    stats_.demodulate_ms = ms_since(t);
  }

  const auto t_bf = Clock::now();
  beamformer_.config() = config_.beamformer;
  Frame frame = beamformer_.beamform(*iq_ptr, grid_);
  stats_.beamform_ms = ms_since(t_bf);

  frame.sequence = frame_count_++;
  stats_.total_ms = ms_since(t_start);
  return frame;
}

std::vector<real> ImagingPipeline::to_display(const Frame& frame, const Medium& medium) {
  const auto t = Clock::now();

  std::vector<real> env = envelope(frame);
  if (config_.apply_tgc) {
    // TGC is applied on the envelope rather than on the channel data because
    // this engine's gain is purely cosmetic: the beamformer already saw the
    // full dynamic range, and boosting deep samples earlier would only amplify
    // the noise along with them.
    real f0 = frame.demodulation_frequency;
    if (f0 <= 0.0f) f0 = 5.0e6f;
    env = apply_tgc(env, frame.grid, medium.attenuation, f0);
  }

  real peak = 0.0f;
  for (real v : env) peak = std::max(peak, v);
  if (peak > 0.0f) {
    const real a = std::clamp(config_.reference_smoothing, 0.0f, 0.999f);
    reference_ = (reference_ > 0.0f) ? (a * reference_ + (1.0f - a) * peak) : peak;
  }

  std::vector<real> img = log_compress(env, config_.dynamic_range_db, config_.gain_db,
                                       reference_ > 0.0f ? reference_ : -1.0f);
  stats_.postprocess_ms += ms_since(t);
  stats_.total_ms += stats_.postprocess_ms;
  return img;
}

std::vector<real> ImagingPipeline::process_display(const ChannelData& data) {
  const Frame frame = process_frame(data);
  return to_display(frame, data.medium);
}

}  // namespace usx
