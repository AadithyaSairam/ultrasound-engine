// usx/doppler.hpp — Quantitative estimators that run on the complex frame data.
//
// This header is the reason the rest of the engine refuses to throw phase away.
// Every estimator here reads the *phase* evolution of the beamformed IQ, either
// across repeated transmits (Doppler: how fast is it moving?) or between two
// frames (speckle tracking: how far did it move?). None of them can be run on a
// log-compressed greyscale image.
//
// Sign convention throughout: positive axial velocity and positive axial
// displacement mean motion in +z, i.e. *away* from the transducer. Clinical
// colour Doppler conventionally paints flow toward the probe red, which is the
// opposite sign; that mapping belongs in the display layer, not here.
#pragma once

#include <vector>

#include "usx/types.hpp"

namespace usx {

// --- Clutter (wall) filtering ----------------------------------------------
//
// Blood scatters ultrasound perhaps 40 dB more weakly than the tissue around
// it, and that tissue is not still — vessel walls pulse, the probe moves, the
// patient breathes. Without removing the slow, overwhelmingly strong tissue
// echo first, a flow estimate returns the motion of the wall, not the blood.
// This one step is the difference between a Doppler image and a coloured-in
// tissue image.
enum class WallFilter : std::uint8_t {
  None = 0,
  // Fit and subtract a low-order polynomial in slow time per pixel. Tissue
  // motion over a short ensemble is smooth and well approximated by a low-order
  // trend; blood is not. Works with very short ensembles, where an FIR filter's
  // transient would eat the whole packet.
  PolynomialRegression = 1,
  // Subtract the ensemble mean only — the cheapest possible clutter rejection,
  // and equivalent to PolynomialRegression of order 0.
  MeanSubtraction = 2,
};

struct DopplerConfig {
  WallFilter wall_filter = WallFilter::PolynomialRegression;
  int polynomial_order = 1;

  // Use the Loupas 2-D autocorrelator, which estimates the true local centre
  // frequency from the axial phase gradient instead of assuming it. This
  // matters because attenuation is frequency-dependent, so the pulse's centre
  // frequency drops with depth; assuming the nominal f0 biases velocity low at
  // depth by several percent. Set false for the classic 1-D Kasai estimator.
  bool loupas = true;

  // Spatial averaging kernel for the autocorrelation, in pixels. Larger kernels
  // reduce estimator variance and blur the flow map — the usual trade.
  int kernel_axial = 4;
  int kernel_lateral = 1;

  // Pixels whose post-wall-filter power falls this far below the frame maximum
  // are marked invalid rather than being given a meaningless velocity.
  real power_threshold_db = -35.0f;

  // Minimum lag-one coherence |R(0,1)| / R(0,0) required before a pixel is
  // called flow.
  //
  // Power alone is a weak discriminator, because what survives the clutter
  // filter over stationary tissue is receive noise, and there can be plenty of
  // it: blood scatters roughly 30 dB more weakly than tissue, so a flow
  // acquisition is far closer to the noise floor than a B-mode one. Noise is
  // white in slow time, so its autocorrelation at lag one is near zero, while
  // real flow is narrowband and highly correlated pulse to pulse. Thresholding
  // on that coherence separates the two even when their powers are comparable —
  // and it is the same quantity as the "variance" output, since
  // variance = 2 * (1 - coherence).
  real coherence_threshold = 0.35f;
};

struct ColorFlowMap {
  ScanGrid grid;
  std::vector<real> velocity;   // m/s, axial component, +ve away from probe
  std::vector<real> power;      // linear power after clutter filtering
  std::vector<real> variance;   // normalized spectral width in [0, 1]
  std::vector<std::uint8_t> valid;

  // Velocity beyond which the phase estimate wraps: v_nyq = c * PRF / (4 * f0).
  // Exceeding it is the familiar colour Doppler aliasing, where the centre of a
  // fast jet flips colour.
  real nyquist_velocity = 0.0f;
};

// Apply the clutter filter to an ensemble of frames in place-equivalent form.
std::vector<Frame> apply_wall_filter(const std::vector<Frame>& ensemble,
                                     WallFilter type, int polynomial_order);

// Colour flow estimation from a slow-time ensemble of beamformed frames.
// `prf` is the effective frame rate of the ensemble (transmits per second
// divided by the number of transmits compounded into each frame).
ColorFlowMap color_flow(const std::vector<Frame>& ensemble, real prf,
                        real speed_of_sound, DopplerConfig config = {});

// Power Doppler: total post-clutter-filter energy, with no velocity estimate.
// Less angle-dependent and far more sensitive than colour flow because it does
// not need a resolvable phase slope — it is what finds slow flow in small
// vessels, at the cost of telling you nothing about direction or speed.
std::vector<real> power_doppler(const std::vector<Frame>& ensemble,
                                WallFilter type = WallFilter::PolynomialRegression,
                                int polynomial_order = 1);

// --- Speckle tracking / elastography ---------------------------------------

struct DisplacementMap {
  ScanGrid grid;
  std::vector<real> axial;        // m, +ve away from probe
  std::vector<real> correlation;  // |normalized cross-correlation| in [0, 1]
};

// Sub-wavelength axial displacement between two frames from the phase of the
// windowed cross-correlation. This is the front half of quasi-static
// elastography: compress the tissue slightly, measure how far each point moved,
// and the spatial derivative of that motion is strain — stiff tissue strains
// less than soft tissue under the same stress.
//
// The phase estimate alone unwraps only within +/- half a wavelength — about
// 150 um at 5 MHz — and beyond that the kernel also decorrelates, so the answer
// is not merely wrapped but meaningless. Since a 1 % compression of a 4 cm deep
// region displaces the bottom by 400 um, that limit is reached immediately in
// any realistic experiment.
//
// `max_lag` fixes it with the standard two-stage estimator: search integer
// sample shifts for the one that maximises correlation, then use the phase at
// that lag for sub-sample refinement. The coarse stage handles the large,
// slowly varying bulk motion and the fine stage supplies the precision, so the
// combination tracks hundreds of microns while still resolving a fraction of
// one. Set it to 0 for phase-only estimation.
DisplacementMap estimate_axial_displacement(const Frame& before, const Frame& after,
                                            real speed_of_sound,
                                            int kernel_axial = 8, int kernel_lateral = 2,
                                            int max_lag = 8);

// Axial strain by least-squares differentiation of the displacement field.
// Differentiating noisy displacement directly amplifies the noise, so a linear
// fit over a window is used instead — the standard least-squares strain
// estimator.
std::vector<real> axial_strain(const DisplacementMap& disp, int fit_length = 15);

}  // namespace usx
