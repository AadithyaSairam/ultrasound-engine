// usx/simulator.hpp — Point-scatterer acoustic forward model.
//
// This is the engine's development harness: it turns a described phantom into
// the raw per-element RF that a real probe would have recorded, so the whole
// pipeline downstream of it can be built, tested and benchmarked with no
// hardware and no downloads — and, crucially, with known ground truth. When you
// can say "there is a point scatterer at exactly (2 mm, 30 mm)", you can
// measure your point spread function instead of squinting at it.
//
// The model is linear and first-order: every scatterer is an independent
// isotropic point re-radiator, echoes superpose, and there is no multiple
// scattering, no non-linear propagation (so no harmonic imaging), no shear
// waves and no aberration from speed-of-sound heterogeneity. Those omissions
// are listed honestly in docs/THEORY.md — they are exactly the effects that
// make real ultrasound harder than simulated ultrasound.
#pragma once

#include <random>
#include <vector>

#include "usx/types.hpp"

namespace usx {

// A single point scatterer. `amplitude` is a dimensionless backscatter
// coefficient: 1.0 is a "typical" tissue scatterer, 0 is anechoic (blood, a
// cyst), and large values are specular reflectors like a needle tip.
struct Scatterer {
  Vec3 position;
  real amplitude = 1.0f;
  Vec3 velocity{0, 0, 0};  // m/s, used to build Doppler ensembles
};

// A collection of scatterers plus the factory helpers that build the standard
// test targets. Real ultrasound phantoms are exactly this: a scattering
// background with known inclusions at known coordinates.
struct Phantom {
  std::vector<Scatterer> scatterers;

  void add(const Scatterer& s) { scatterers.push_back(s); }
  std::size_t size() const { return scatterers.size(); }

  // Isolated point targets. The image of one of these *is* the point spread
  // function, so this is the phantom used to measure resolution.
  static Phantom point_targets(const std::vector<Vec3>& positions, real amplitude = 1.0f);

  // A block of uniformly random scatterers. When enough of them fall inside one
  // resolution cell their echoes interfere randomly, producing the granular
  // texture called speckle. Speckle is not noise — it is a deterministic
  // interference pattern, which is why it can be tracked to measure motion and
  // strain, and why simply averaging it away is throwing information out.
  static Phantom speckle_box(real x_min, real x_max, real z_min, real z_max,
                             real y_half_thickness, int n_scatterers,
                             std::uint32_t seed = 1234, real amplitude_sigma = 1.0f);

  // How many scatterers a box needs for fully developed speckle: the rule of
  // thumb is >= 10 per resolution cell.
  static int recommended_scatterer_count(const Probe& probe, const Medium& medium,
                                         real x_span, real z_span, real y_span,
                                         real f_number = 2.0f,
                                         real per_cell = 10.0f);

  // Scale the amplitude of every scatterer inside a circle. `scale` of 0 makes
  // an anechoic cyst, <1 a hypoechoic lesion, >1 a hyperechoic one. Contrast
  // targets like these are how CNR/gCNR are measured.
  void scale_amplitude_in_circle(const Vec3& center, real radius, real scale);

  // Add a straight vessel of moving scatterers with a parabolic (laminar) flow
  // profile — the standard test target for colour and spectral Doppler.
  // `axis_angle` is measured from the x axis in the imaging plane, so an angle
  // of 0 is pure lateral flow, which produces *no* axial Doppler shift at all.
  void add_flow_tube(const Vec3& center, real radius, real length, real axis_angle,
                     real peak_velocity, int n_scatterers, std::uint32_t seed = 99,
                     real amplitude = 0.05f);
};

struct SimulatorConfig {
  // Two-way pulse shape. The transducer's finite bandwidth is what gives the
  // pulse its length, and the pulse length is what sets axial resolution.
  real pulse_cycles = 2.5f;   // used only if bandwidth <= 0

  Window tx_window = Window::Hann;
  real tx_f_number = 2.0f;    // active aperture for Focused transmits

  // Internal oversampling of the incident waveform, so that the sub-sample time
  // shifts applied on receive interpolate accurately.
  int oversampling = 4;

  // Model the true transmit field by summing every element's contribution
  // (diffraction, edge waves, finite aperture). Turning this off collapses the
  // transmit field to a single geometric arrival per scatterer: far faster,
  // and adequate for large speckle phantoms where the detail is washed out
  // anyway, but it will not reproduce transmit sidelobes.
  bool full_transmit_diffraction = true;

  // Discard the tails of the per-scatterer incident waveform below this level
  // relative to its peak. This is the main speed/accuracy dial when
  // full_transmit_diffraction is on.
  real incident_trim_db = -40.0f;
  real max_incident_window_us = 30.0f;

  bool apply_attenuation = true;
  bool apply_directivity = true;
  bool apply_spreading = true;

  // Additive white Gaussian receive noise, in dB relative to the peak signal of
  // the event. Set to a very negative number to disable. Thermal noise is what
  // ultimately limits imaging depth, so a noiseless simulation flatters every
  // beamformer equally and hides the differences between them.
  real noise_db = -60.0f;
  std::uint32_t noise_seed = 7;

  // Advance scatterer positions by their velocity between transmit events,
  // using the acquisition PRF. Required for Doppler; harmless otherwise.
  bool enable_motion = true;
};

class Simulator {
 public:
  Simulator(Probe probe, Medium medium, Acquisition acq, SimulatorConfig config = {});

  // Build a set of transmit events with delay laws already computed.
  std::vector<Transmit> make_plane_waves(const std::vector<real>& angles) const;
  std::vector<Transmit> make_focused_scan(int n_lines, real x_min, real x_max,
                                          real focal_depth, real angle = 0.0f) const;
  std::vector<Transmit> make_diverging_waves(const std::vector<real>& angles,
                                             real virtual_source_depth) const;

  // Repeat a transmit sequence `n_ensemble` times back-to-back. This is how a
  // Doppler acquisition is built: the same look, fired repeatedly, so that the
  // phase change between repetitions reveals motion.
  static std::vector<Transmit> repeat(const std::vector<Transmit>& txs, int n_ensemble);

  // Run the forward model. The returned ChannelData carries the probe, medium,
  // acquisition and transmit list along with the samples, so a beamformer
  // downstream needs no side-channel metadata to interpret it.
  ChannelData simulate(const std::vector<Transmit>& transmits, const Phantom& phantom) const;

  const Probe& probe() const { return probe_; }
  const Medium& medium() const { return medium_; }
  const Acquisition& acquisition() const { return acq_; }
  const SimulatorConfig& config() const { return config_; }
  Acquisition& acquisition() { return acq_; }

  // The two-way pulse this simulator uses, sampled at fs * oversampling.
  // Exposed because matched filtering and axial-resolution tests need it.
  std::vector<real> pulse(real* half_duration = nullptr) const;

 private:
  Probe probe_;
  Medium medium_;
  Acquisition acq_;
  SimulatorConfig config_;
};

}  // namespace usx
