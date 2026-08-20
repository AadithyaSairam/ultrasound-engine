// usx/types.hpp — Core data model for the ultrasound imaging engine.
//
// Everything in this engine is expressed in SI units (metres, seconds, hertz)
// unless a name says otherwise. Sticking to one unit system removes an entire
// category of bug: in ultrasound you are constantly converting between time,
// distance and sample index, and mixing mm with m silently halves your image
// depth.
//
// The three coordinate conventions used throughout:
//
//   x  — lateral   (along the transducer face, 0 at the array centre)
//   y  — elevation (out of the imaging plane; this engine is 2-D so y == 0)
//   z  — axial     (depth into the tissue, 0 at the transducer face, +ve down)
//
//   t  — acquisition time, with t = 0 defined as the instant the transmitted
//        wavefront passes through the origin (0, 0, 0). Every delay law in the
//        engine is written to make that statement true, so that a beamformer
//        and a simulator that agree on geometry automatically agree on timing.
//        See docs/THEORY.md, "The t = 0 convention".
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace usx {

using real = float;
using cplx = std::complex<real>;

constexpr real kPi = static_cast<real>(3.14159265358979323846);

// ---------------------------------------------------------------------------
// Vec3 — a tiny 3-vector. Deliberately not a library dependency; the engine
// only ever needs distance, so a full linear-algebra package would be dead
// weight in the inner loop.
// ---------------------------------------------------------------------------
struct Vec3 {
  real x = 0, y = 0, z = 0;

  Vec3() = default;
  Vec3(real x_, real y_, real z_) : x(x_), y(y_), z(z_) {}

  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator*(real s) const { return {x * s, y * s, z * s}; }

  real norm() const { return std::sqrt(x * x + y * y + z * z); }
  real norm_sq() const { return x * x + y * y + z * z; }
};

inline real distance(const Vec3& a, const Vec3& b) { return (a - b).norm(); }

// ---------------------------------------------------------------------------
// Probe — a linear/phased array of rectangular elements along x.
//
// Element pitch is the centre-to-centre spacing; element width is the physical
// active width. The gap between them (the kerf) is what stops elements from
// cross-talking. Pitch matters for grating lobes: if pitch > lambda/2 you get
// copies of the mainlobe at large steering angles, which is why phased arrays
// (which steer a lot) are built at lambda/2 and linear arrays (which barely
// steer) can get away with ~lambda.
// ---------------------------------------------------------------------------
struct Probe {
  int n_elements = 128;
  real pitch = 300e-6f;             // m, centre-to-centre element spacing
  real element_width = 270e-6f;     // m, active width of one element
  real elevation_height = 5e-3f;    // m, fixed elevation aperture
  real center_frequency = 5.0e6f;   // Hz, transducer centre frequency f0
  real bandwidth = 0.6f;            // fractional -6 dB bandwidth (0..1+)

  // Position of element i, centred on x = 0.
  Vec3 element_position(int i) const {
    const real x = (static_cast<real>(i) - 0.5f * (n_elements - 1)) * pitch;
    return {x, 0.0f, 0.0f};
  }

  real aperture_width() const {
    return static_cast<real>(n_elements - 1) * pitch + element_width;
  }

  real wavelength(real speed_of_sound) const {
    return speed_of_sound / center_frequency;
  }

  void validate() const;
};

// ---------------------------------------------------------------------------
// Medium — the acoustic properties the engine assumes about the tissue.
//
// A real body is not homogeneous, and assuming it is, is the single largest
// source of image degradation in clinical ultrasound: the beamformer computes
// delays from an assumed speed of sound (conventionally 1540 m/s), and fat at
// ~1450 m/s or muscle at ~1580 m/s makes those delays wrong, which defocuses
// the image. Estimating the true value is one of the quantitative-imaging
// tasks this engine is designed to feed (see docs/THEORY.md).
// ---------------------------------------------------------------------------
struct Medium {
  real speed_of_sound = 1540.0f;        // m/s
  real attenuation = 0.5f;              // dB / (cm * MHz), one-way
  real density = 1000.0f;               // kg/m^3 (unused by the linear model)

  void validate() const;
};

// ---------------------------------------------------------------------------
// Acquisition — the receive-side sampling settings.
// ---------------------------------------------------------------------------
struct Acquisition {
  real sampling_frequency = 40.0e6f;  // Hz, RF sample rate
  int n_samples = 2048;               // samples recorded per element per event
  real t0 = 0.0f;                     // acquisition time of sample index 0 (s)
  real prf = 5000.0f;                 // Hz, pulse repetition frequency

  real sample_time(int n) const {
    return t0 + static_cast<real>(n) / sampling_frequency;
  }

  // Inverse of sample_time: the (fractional) sample index of a given time.
  real time_to_sample(real t) const {
    return (t - t0) * sampling_frequency;
  }

  // Maximum unambiguous depth: any echo arriving after the next pulse is fired
  // is aliased onto the following acquisition.
  real max_depth(real c) const { return 0.5f * c / prf; }

  void validate() const;
};

// ---------------------------------------------------------------------------
// Transmit — one transmit event (one "shot").
//
// This engine supports two transmit geometries, which between them cover the
// overwhelming majority of B-mode imaging:
//
//   Focused    — delays curve the wavefront to converge on a focal point.
//                Classic line-by-line imaging: one transmit per scanline, so a
//                128-line image costs 128 transmits and the frame rate is
//                capped by the round-trip time times the line count.
//
//   PlaneWave  — a single flat (optionally steered) wavefront insonifies the
//                whole field of view at once. One transmit reconstructs an
//                entire image, so frame rates jump by two orders of magnitude
//                ("ultrafast" imaging), at the cost of image quality — which is
//                recovered by coherently compounding several steered angles.
//
//   Diverging  — a virtual source *behind* the array, spreading the beam out to
//                cover a sector. Same maths as Focused with a negative-z focus;
//                this is how phased arrays do ultrafast imaging.
// ---------------------------------------------------------------------------
enum class TransmitType : std::uint8_t { Focused = 0, PlaneWave = 1, Diverging = 2 };

struct Transmit {
  TransmitType type = TransmitType::PlaneWave;

  real angle = 0.0f;          // rad, steering angle from the z axis (+ve = +x)
  Vec3 focus{0, 0, 0.03f};    // m, focal point (Focused / Diverging only)
  Vec3 origin{0, 0, 0};       // m, lateral origin of this beam / scanline

  // Per-element transmit delays and apodization, length == probe.n_elements.
  // A weight of exactly 0 means the element does not fire.
  std::vector<real> delays;
  std::vector<real> apodization;

  // Number of elements with non-zero weight (informational).
  int n_active() const;
};

// ---------------------------------------------------------------------------
// ChannelData — raw pre-beamforming data: what the receive electronics
// actually hand you. This is the engine's input, whether it came from the
// simulator, a stored dataset, or a live scanner.
//
// Layout is [event][channel][sample], contiguous with sample varying fastest,
// because every consumer of this data streams along the sample axis.
//
// The buffer is real-valued RF, or complex IQ once demodulated. Both are stored
// in the same struct with `is_iq` distinguishing them, so the beamformer can
// take either and apply phase rotation only when it must.
// ---------------------------------------------------------------------------
struct ChannelData {
  Probe probe;
  Medium medium;
  Acquisition acq;
  std::vector<Transmit> transmits;

  bool is_iq = false;
  // RF storage (is_iq == false)
  std::vector<real> rf;
  // IQ storage (is_iq == true); demodulation_frequency is the frequency the
  // baseband signal was mixed down from — the beamformer needs it to convert a
  // sub-sample time shift into a phase rotation.
  std::vector<cplx> iq;
  real demodulation_frequency = 0.0f;

  int n_events() const { return static_cast<int>(transmits.size()); }
  int n_channels() const { return probe.n_elements; }
  int n_samples() const { return acq.n_samples; }

  std::size_t index(int event, int channel, int sample) const {
    return (static_cast<std::size_t>(event) * n_channels() + channel) *
               static_cast<std::size_t>(n_samples()) + sample;
  }

  std::size_t size() const {
    return static_cast<std::size_t>(n_events()) * n_channels() * n_samples();
  }

  void allocate();
  void validate() const;
};

// ---------------------------------------------------------------------------
// ScanGrid — where in space the beamformer reconstructs pixels.
//
// Two flavours:
//   Cartesian — a rectangular x/z raster. Natural for linear arrays, and needs
//               no scan conversion because the pixels already are the display.
//   Sector    — a polar theta/r raster from an apex. Natural for phased and
//               curvilinear arrays; must be scan-converted to a raster before
//               it can be displayed.
//
// Storing the grid as an explicit list of pixel positions (rather than as a
// formula) is what lets one beamformer serve both, and lets a caller supply a
// completely arbitrary sampling — including a single line, or a scattered set
// of points for a quantitative measurement.
// ---------------------------------------------------------------------------
enum class GridType : std::uint8_t { Cartesian = 0, Sector = 1 };

struct ScanGrid {
  GridType type = GridType::Cartesian;

  // Axis 0 is the "slow"/lateral axis (x, or theta); axis 1 is the "fast"/axial
  // axis (z, or r). Pixels are stored [axis0][axis1] with axis1 fastest.
  int n_axis0 = 0;
  int n_axis1 = 0;

  std::vector<real> axis0;  // m (Cartesian x) or rad (Sector theta)
  std::vector<real> axis1;  // m (depth z, or range r)
  Vec3 apex{0, 0, 0};       // sector apex; ignored for Cartesian

  std::vector<Vec3> points; // length n_axis0 * n_axis1, axis1 fastest

  int n_pixels() const { return n_axis0 * n_axis1; }

  static ScanGrid cartesian(real x_min, real x_max, int nx,
                            real z_min, real z_max, int nz);
  static ScanGrid sector(real theta_min, real theta_max, int n_theta,
                         real r_min, real r_max, int n_r, Vec3 apex);

  void build_points();
  void validate() const;
};

// ---------------------------------------------------------------------------
// Frame — the engine's canonical output: a beamformed *complex* image.
//
// This is deliberately not a picture. Log-compressed 8-bit greyscale is the end
// of the line: once you have thrown away phase you cannot do Doppler, you
// cannot do elastography by speckle tracking, you cannot coherently compound,
// and you cannot estimate speed of sound. So the engine's product is the
// complex IQ image, and display is a lossy consumer of it downstream.
// ---------------------------------------------------------------------------
struct Frame {
  ScanGrid grid;
  std::vector<cplx> data;   // n_pixels(), axis1 fastest
  real demodulation_frequency = 0.0f;
  double timestamp = 0.0;   // s, acquisition time of the frame
  std::int64_t sequence = 0;

  void allocate() { data.assign(static_cast<std::size_t>(grid.n_pixels()), cplx(0, 0)); }
  const cplx& at(int i0, int i1) const { return data[static_cast<std::size_t>(i0) * grid.n_axis1 + i1]; }
  cplx& at(int i0, int i1) { return data[static_cast<std::size_t>(i0) * grid.n_axis1 + i1]; }
};

// ---------------------------------------------------------------------------
// Apodization windows, used on both transmit and receive apertures. Tapering
// the aperture trades mainlobe width (resolution) for sidelobe level
// (contrast) — the same trade-off as windowing an FFT, because it is the same
// mathematics: the beam pattern is the spatial Fourier transform of the
// aperture weighting.
// ---------------------------------------------------------------------------
enum class Window : std::uint8_t { Rect = 0, Hann = 1, Hamming = 2, Tukey = 3, Blackman = 4 };

// Evaluate a window at normalized position u in [-1, 1] across the aperture.
real window_value(Window w, real u, real tukey_alpha = 0.5f);

Window window_from_string(const std::string& name);

}  // namespace usx
