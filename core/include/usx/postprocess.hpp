// usx/postprocess.hpp — From complex beamformed data to something displayable.
//
// This is the lossy end of the pipeline, and it is deliberately the *last*
// stage. Everything upstream keeps complex IQ, because phase is what Doppler,
// speckle tracking and coherent compounding are built on. Only here is it
// thrown away, and only for the display path.
//
// The three steps below are also, historically, where "the image looks better"
// gets confused with "the image is better": time gain compensation, dynamic
// range and gain are cosmetic controls that change nothing measurable. Any
// quantitative claim should be made on the data upstream of this header.
#pragma once

#include <vector>

#include "usx/types.hpp"

namespace usx {

// Envelope (magnitude) of a beamformed frame.
//
// If the frame is complex baseband (demodulation_frequency > 0) this is simply
// the modulus. If the frame was beamformed straight from RF it is real-valued
// and oscillating at f0, so the analytic signal is formed first with a Hilbert
// transform along the axial direction; taking |real| instead would produce a
// rectified image with the carrier still in it.
std::vector<real> envelope(const Frame& frame);

// Time gain compensation: undo the depth-dependent attenuation so that equally
// scattering tissue looks equally bright at all depths. `attenuation` is in
// dB/(cm*MHz) one-way, matching Medium.
std::vector<real> apply_tgc(const std::vector<real>& env, const ScanGrid& grid,
                            real attenuation_db_cm_mhz, real center_frequency_hz);

// Log compression to a displayable [0, 1] image.
//
// Echo amplitudes in tissue span five or six orders of magnitude, far beyond
// what a display or an eye resolves linearly, so the standard is to map a
// dynamic range window (typically 50-60 dB below the peak) onto the greyscale.
// `reference` <= 0 means "use the frame maximum".
std::vector<real> log_compress(const std::vector<real>& env, real dynamic_range_db = 60.0f,
                              real gain_db = 0.0f, real reference = -1.0f);

// Resample a polar (sector) image onto a Cartesian raster for display.
// Returns the raster plus the physical extent it covers.
struct ScanConverted {
  int width = 0;
  int height = 0;
  real x_min = 0, x_max = 0, z_min = 0, z_max = 0;
  std::vector<real> pixels;  // height * width, row-major, NaN outside the sector
};

ScanConverted scan_convert(const std::vector<real>& image, const ScanGrid& grid,
                           int width, int height, real fill_value = 0.0f);

// Discrete Hilbert transform along the fast (axial) axis, giving the analytic
// signal of a real-valued frame. Exposed because RF-domain processing steps
// outside this header need it too.
std::vector<cplx> analytic_along_axis1(const std::vector<real>& data, int n_axis0, int n_axis1);

}  // namespace usx
