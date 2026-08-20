// core/tests/test_main.cpp — Correctness tests for the usx core.
//
// These are not smoke tests. Each one pins down a quantity that has a known
// right answer from geometry or from the phantom that was simulated, because
// the failure mode of a beamformer is not a crash — it is an image that looks
// plausible and is wrong. A point target placed at 30 mm that reconstructs at
// 29.6 mm still looks like an ultrasound image.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "usx/beamformer.hpp"
#include "usx/demodulate.hpp"
#include "usx/doppler.hpp"
#include "usx/geometry.hpp"
#include "usx/pipeline.hpp"
#include "usx/postprocess.hpp"
#include "usx/simulator.hpp"

using namespace usx;

namespace {

int g_failures = 0;
int g_checks = 0;
std::string g_current;

void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL [%s] %s\n", g_current.c_str(), what.c_str());
  }
}

void check_near(double got, double want, double tol, const std::string& what) {
  ++g_checks;
  if (!(std::abs(got - want) <= tol)) {
    ++g_failures;
    std::printf("  FAIL [%s] %s: got %.6g, want %.6g +/- %.3g\n", g_current.c_str(),
                what.c_str(), got, want, tol);
  }
}

struct Section {
  explicit Section(const char* name) {
    g_current = name;
    std::printf("%s\n", name);
  }
};

// A standard 5 MHz linear array, used by most tests below.
Probe standard_probe() {
  Probe p;
  p.n_elements = 96;
  p.pitch = 300e-6f;
  p.element_width = 270e-6f;
  p.center_frequency = 5.0e6f;
  p.bandwidth = 0.6f;
  return p;
}

Acquisition standard_acq(int n_samples = 2600) {
  Acquisition a;
  a.sampling_frequency = 40.0e6f;
  a.n_samples = n_samples;
  a.t0 = 0.0f;
  a.prf = 4000.0f;
  return a;
}

// Locate the peak of |frame| and return its grid coordinates.
void peak_location(const Frame& f, real* x, real* z, real* mag) {
  double best = -1;
  int bi = 0, bj = 0;
  for (int i = 0; i < f.grid.n_axis0; ++i)
    for (int j = 0; j < f.grid.n_axis1; ++j) {
      const double m = std::abs(f.at(i, j));
      if (m > best) { best = m; bi = i; bj = j; }
    }
  *x = f.grid.axis0[static_cast<std::size_t>(bi)];
  *z = f.grid.axis1[static_cast<std::size_t>(bj)];
  *mag = static_cast<real>(best);
}

// Lateral full width at half maximum, in metres, of the row through the peak.
// The half-power crossings are interpolated between pixels, so the result is not
// quantized to the grid and the combiners can be compared meaningfully.
real lateral_fwhm(const Frame& f) {
  double best = -1;
  int bi = 0, bj = 0;
  for (int i = 0; i < f.grid.n_axis0; ++i)
    for (int j = 0; j < f.grid.n_axis1; ++j) {
      const double m = std::abs(f.at(i, j));
      if (m > best) { best = m; bi = i; bj = j; }
    }
  const double half = best * 0.5;
  const auto val = [&](int i) { return static_cast<double>(std::abs(f.at(i, bj))); };
  const auto pos = [&](int i) { return static_cast<double>(f.grid.axis0[static_cast<std::size_t>(i)]); };

  int lo = bi;
  while (lo > 0 && val(lo) > half) --lo;
  double x_lo = pos(lo);
  if (lo < bi && val(lo) <= half) {
    const double t = (half - val(lo)) / (val(lo + 1) - val(lo));
    x_lo = pos(lo) + t * (pos(lo + 1) - pos(lo));
  }

  int hi = bi;
  while (hi < f.grid.n_axis0 - 1 && val(hi) > half) ++hi;
  double x_hi = pos(hi);
  if (hi > bi && val(hi) <= half) {
    const double t = (half - val(hi)) / (val(hi - 1) - val(hi));
    x_hi = pos(hi) + t * (pos(hi - 1) - pos(hi));
  }
  return static_cast<real>(x_hi - x_lo);
}

}  // namespace

// ---------------------------------------------------------------------------

void test_windows() {
  Section s("windows and apodization");
  check_near(window_value(Window::Hann, 0.0f), 1.0, 1e-6, "Hann peak is 1");
  check_near(window_value(Window::Hann, -1.0f), 0.0, 1e-6, "Hann is 0 at the left edge");
  check_near(window_value(Window::Hann, 1.0f), 0.0, 1e-6, "Hann is 0 at the right edge");
  check_near(window_value(Window::Rect, 0.5f), 1.0, 1e-6, "Rect is flat");
  check_near(window_value(Window::Rect, 1.5f), 0.0, 1e-6, "windows vanish outside the aperture");
  check(window_value(Window::Tukey, 0.0f, 0.5f) == 1.0f, "Tukey is flat in the middle");
}

void test_transmit_geometry() {
  Section s("transmit delay laws");
  const Probe probe = standard_probe();
  const Medium med;
  const real c = med.speed_of_sound;

  // A focused transmit must make every element's contribution land on the focus
  // at the same instant. This is the definition of focusing, and if it fails
  // nothing downstream can be right.
  Transmit tx;
  tx.type = TransmitType::Focused;
  tx.origin = Vec3(0, 0, 0);
  tx.focus = Vec3(0, 0, 0.030f);
  compute_transmit_delays(probe, med, tx, Window::Rect, 2.0f, nullptr);

  real t_ref = -1;
  bool ok = true;
  for (int i = 0; i < probe.n_elements; ++i) {
    if (tx.apodization[static_cast<std::size_t>(i)] == 0.0f) continue;
    const real t = tx.delays[static_cast<std::size_t>(i)] +
                   distance(tx.focus, probe.element_position(i)) / c;
    if (t_ref < 0) t_ref = t;
    // Tolerance is set by float32 precision at ~2e-5 s, not by the algorithm:
    // 50 ps is a quarter of a thousandth of a 5 MHz period.
    else if (std::abs(t - t_ref) > 5e-11f) ok = false;
  }
  check(ok, "all focused-transmit contributions arrive at the focus simultaneously");
  check_near(transmit_arrival_time(tx, tx.focus, c), t_ref, 1e-9,
             "the wavefront model agrees with the delay law at the focus");
  check_near(transmit_arrival_time(tx, Vec3(0, 0, 0), c), 0.0, 1e-9,
             "focused: t = 0 at the beam origin");

  // A steered plane wave must reach a point at exactly the projection of that
  // point onto the wavefront normal, divided by c.
  Transmit pw;
  pw.type = TransmitType::PlaneWave;
  pw.angle = 0.20f;
  compute_transmit_delays(probe, med, pw, Window::Rect, 2.0f, nullptr);
  const Vec3 p(0.004f, 0, 0.025f);
  const real expect = (p.x * std::sin(pw.angle) + p.z * std::cos(pw.angle)) / c;
  check_near(transmit_arrival_time(pw, p, c), expect, 1e-12, "steered plane-wave arrival time");
  check_near(transmit_arrival_time(pw, Vec3(0, 0, 0), c), 0.0, 1e-12,
             "plane wave: t = 0 at the origin");

  // The plane-wave delay law and the wavefront model must agree at the array
  // itself, or the simulator and the beamformer are using different clocks.
  bool consistent = true;
  for (int i = 0; i < probe.n_elements; ++i) {
    const Vec3 pe = probe.element_position(i);
    if (std::abs(pw.delays[static_cast<std::size_t>(i)] - transmit_arrival_time(pw, pe, c)) > 1e-12f)
      consistent = false;
  }
  check(consistent, "plane-wave delay law matches the wavefront model at the elements");

  // Diverging wave: a virtual source behind the array.
  Transmit dv;
  dv.type = TransmitType::Diverging;
  dv.focus = Vec3(0, 0, -0.010f);
  compute_transmit_delays(probe, med, dv, Window::Rect, 2.0f, nullptr);
  check_near(transmit_arrival_time(dv, Vec3(0, 0, 0), c), 0.0, 1e-9,
             "diverging: t = 0 at the beam origin");
  check(transmit_arrival_time(dv, Vec3(0, 0, 0.030f), c) > 0, "diverging wave propagates forward");
}

void test_directivity() {
  Section s("element directivity");
  const real lambda = 1540.0f / 5.0e6f;
  check_near(element_directivity(270e-6f, lambda, 0.0f), 1.0, 1e-5, "on-axis response is unity");
  check(element_directivity(270e-6f, lambda, 0.8f) < element_directivity(270e-6f, lambda, 0.2f),
        "response falls off with angle");
  // A narrower element is less directional at the same angle.
  check(element_directivity(100e-6f, lambda, 0.7f) > element_directivity(300e-6f, lambda, 0.7f),
        "narrower elements are more omnidirectional");
}

void test_demodulation() {
  Section s("RF to IQ demodulation");
  Probe probe = standard_probe();
  probe.n_elements = 4;
  Medium med;
  Acquisition acq = standard_acq(512);

  ChannelData cd;
  cd.probe = probe;
  cd.medium = med;
  cd.acq = acq;
  Transmit tx;
  tx.type = TransmitType::PlaneWave;
  compute_transmit_delays(probe, med, tx, Window::Rect, 2.0f, nullptr);
  cd.transmits = {tx};
  cd.is_iq = false;
  cd.allocate();

  // A pure tone at f0 with unit amplitude should demodulate to a constant
  // baseband signal of magnitude 1 (the factor-of-2 restoration in demodulate()
  // is what makes this true rather than 0.5).
  const real f0 = probe.center_frequency;
  const real amp = 1.0f;
  for (int ch = 0; ch < probe.n_elements; ++ch)
    for (int n = 0; n < acq.n_samples; ++n)
      cd.rf[cd.index(0, ch, n)] = amp * std::cos(2.0f * kPi * f0 * acq.sample_time(n));

  DemodConfig dc;
  dc.decimation = 1;
  const ChannelData iq = demodulate(cd, dc);
  check(iq.is_iq, "demodulate produces IQ");
  check_near(iq.demodulation_frequency, f0, 1.0, "demodulation frequency is recorded");

  double mean = 0;
  int count = 0;
  for (int n = 64; n < acq.n_samples - 64; ++n) {
    mean += std::abs(iq.iq[iq.index(0, 0, n)]);
    ++count;
  }
  mean /= count;
  check_near(mean, amp, 0.02, "IQ magnitude equals the RF amplitude");

  // Decimation must not move the time axis: sample k of the decimated signal
  // still corresponds to t0 + k*dec/fs.
  DemodConfig dc4;
  dc4.decimation = 4;
  const ChannelData iq4 = demodulate(cd, dc4);
  check_near(iq4.acq.sampling_frequency, acq.sampling_frequency / 4, 1.0,
             "decimation lowers the sample rate");
  check(iq4.acq.n_samples == (acq.n_samples + 3) / 4, "decimation shortens the record");
}

void test_point_target_localization() {
  Section s("point target localization");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  SimulatorConfig cfg;
  cfg.noise_db = -200.0f;
  Simulator sim(probe, med, acq, cfg);

  const std::vector<Vec3> targets = {Vec3(0.000f, 0, 0.015f), Vec3(0.006f, 0, 0.030f),
                                     Vec3(-0.008f, 0, 0.045f)};
  const Phantom ph = Phantom::point_targets(targets);

  // Both transmit schemes must put the target in the same place. They use
  // completely different delay laws, so agreement between them is a real check.
  struct Case { const char* name; std::vector<Transmit> txs; Compounding comp; };
  std::vector<Case> cases;
  cases.push_back({"plane wave x1", sim.make_plane_waves({0.0f}), Compounding::Coherent});
  cases.push_back({"plane wave x11",
                   sim.make_plane_waves({-0.16f, -0.128f, -0.096f, -0.064f, -0.032f, 0.0f, 0.032f,
                                         0.064f, 0.096f, 0.128f, 0.16f}),
                   Compounding::Coherent});
  cases.push_back({"focused scan", sim.make_focused_scan(48, -0.012f, 0.012f, 0.030f),
                   Compounding::Select});

  for (const Case& cs : cases) {
    const ChannelData raw = sim.simulate(cs.txs, ph);
    const ChannelData iq = demodulate(raw, {});
    BeamformerConfig bc;
    bc.compounding = cs.comp;
    Beamformer bf(bc);

    for (const Vec3& t : targets) {
      // A tight grid around each target: 0.1 mm pixels, so the tolerance below
      // is genuinely testing the reconstruction and not the grid.
      const ScanGrid grid = ScanGrid::cartesian(t.x - 0.004f, t.x + 0.004f, 81,
                                                t.z - 0.004f, t.z + 0.004f, 81);
      const Frame f = bf.beamform(iq, grid);
      real px, pz, mag;
      peak_location(f, &px, &pz, &mag);
      check_near(px, t.x, 2.1e-4, std::string(cs.name) + ": lateral position of target at z=" +
                                     std::to_string(t.z * 1000) + "mm");
      check_near(pz, t.z, 2.1e-4, std::string(cs.name) + ": axial position of target at z=" +
                                      std::to_string(t.z * 1000) + "mm");
      check(mag > 0, std::string(cs.name) + ": target has non-zero amplitude");
    }
  }
}

void test_rf_and_iq_paths_agree() {
  Section s("RF and IQ beamforming paths agree");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  SimulatorConfig cfg;
  cfg.noise_db = -200.0f;
  Simulator sim(probe, med, acq, cfg);

  const Phantom ph = Phantom::point_targets({Vec3(0.003f, 0, 0.025f)});
  const ChannelData raw = sim.simulate(sim.make_plane_waves({0.0f}), ph);
  const ChannelData iq = demodulate(raw, {});
  const ScanGrid grid = ScanGrid::cartesian(-0.004f, 0.010f, 141, 0.020f, 0.030f, 201);

  Beamformer bf;
  const Frame f_rf = bf.beamform(raw, grid);
  const Frame f_iq = bf.beamform(iq, grid);

  real x1, z1, m1, x2, z2, m2;
  peak_location(f_rf, &x1, &z1, &m1);
  peak_location(f_iq, &x2, &z2, &m2);
  check_near(x1, x2, 1.1e-4, "RF and IQ paths find the same lateral position");
  check_near(z1, z2, 1.1e-4, "RF and IQ paths find the same axial position");

  // Envelope detection must work on both, via Hilbert for RF and modulus for IQ.
  const std::vector<real> e_rf = envelope(f_rf);
  const std::vector<real> e_iq = envelope(f_iq);
  double pk_rf = 0, pk_iq = 0;
  for (real v : e_rf) pk_rf = std::max(pk_rf, static_cast<double>(v));
  for (real v : e_iq) pk_iq = std::max(pk_iq, static_cast<double>(v));
  check(pk_rf > 0 && pk_iq > 0, "both envelopes are non-degenerate");
}

void test_combiners() {
  Section s("adaptive beamformers");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  SimulatorConfig cfg;
  cfg.noise_db = -200.0f;
  Simulator sim(probe, med, acq, cfg);

  const Phantom ph = Phantom::point_targets({Vec3(0.0f, 0, 0.030f)});
  const ChannelData iq = demodulate(sim.simulate(sim.make_plane_waves({0.0f}), ph), {});
  const ScanGrid grid = ScanGrid::cartesian(-0.004f, 0.004f, 161, 0.028f, 0.032f, 81);

  BeamformerConfig bc;
  bc.f_number = 1.5f;

  bc.combiner = Combiner::DAS;
  const Frame f_das = Beamformer(bc).beamform(iq, grid);
  bc.combiner = Combiner::DMAS;
  const Frame f_dmas = Beamformer(bc).beamform(iq, grid);
  bc.combiner = Combiner::MV;
  bc.mv_subarray = 16;
  bc.mv_diagonal_loading = 1e-2f;
  const Frame f_mv = Beamformer(bc).beamform(iq, grid);

  real x, z, m;
  peak_location(f_das, &x, &z, &m);
  check_near(x, 0.0, 1.1e-4, "DAS localizes the target laterally");
  check_near(z, 0.030, 1.1e-4, "DAS localizes the target axially");
  peak_location(f_dmas, &x, &z, &m);
  check_near(x, 0.0, 1.1e-4, "DMAS localizes the target laterally");
  check_near(z, 0.030, 1.1e-4, "DMAS localizes the target axially");
  peak_location(f_mv, &x, &z, &m);
  check_near(x, 0.0, 1.1e-4, "MV localizes the target laterally");
  check_near(z, 0.030, 1.1e-4, "MV localizes the target axially");

  // The whole reason to pay for the adaptive combiners is a narrower mainlobe.
  // If they are not narrower than DAS on a clean point target, they are broken.
  const real w_das = lateral_fwhm(f_das);
  const real w_dmas = lateral_fwhm(f_dmas);
  const real w_mv = lateral_fwhm(f_mv);
  std::printf("  (lateral FWHM: DAS %.3f mm, DMAS %.3f mm, MV %.3f mm)\n", w_das * 1e3,
              w_dmas * 1e3, w_mv * 1e3);
  check(w_dmas < w_das * 0.95f, "DMAS resolves finer than DAS");
  check(w_mv < w_das * 0.95f, "MV resolves finer than DAS");

  // Coherence factor weighting must not move the target.
  bc.combiner = Combiner::DAS;
  bc.coherence_factor = true;
  const Frame f_cf = Beamformer(bc).beamform(iq, grid);
  peak_location(f_cf, &x, &z, &m);
  check_near(x, 0.0, 1.1e-4, "coherence factor preserves lateral position");
}

void test_resolution_scales_with_aperture() {
  Section s("resolution physics");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  SimulatorConfig cfg;
  cfg.noise_db = -200.0f;
  Simulator sim(probe, med, acq, cfg);

  const Phantom ph = Phantom::point_targets({Vec3(0.0f, 0, 0.030f)});
  const ChannelData iq = demodulate(sim.simulate(sim.make_plane_waves({0.0f}), ph), {});
  const ScanGrid grid = ScanGrid::cartesian(-0.005f, 0.005f, 201, 0.029f, 0.031f, 41);

  // Lateral resolution is ~ f_number * lambda. Halving the f-number (opening
  // the aperture) must roughly halve the beam width; this is the single most
  // important scaling law in the whole subject, so it gets its own test.
  BeamformerConfig bc;
  bc.f_number = 3.0f;
  const real w3 = lateral_fwhm(Beamformer(bc).beamform(iq, grid));
  bc.f_number = 1.5f;
  const real w15 = lateral_fwhm(Beamformer(bc).beamform(iq, grid));
  const real lambda = probe.wavelength(med.speed_of_sound);
  std::printf("  (FWHM at F/3 = %.3f mm = %.2f lambda; at F/1.5 = %.3f mm = %.2f lambda)\n",
              w3 * 1e3, w3 / lambda, w15 * 1e3, w15 / lambda);
  check(w15 < w3, "a wider aperture resolves better");
  check(w15 < w3 * 0.75f, "halving the f-number substantially narrows the beam");
  // A single unsteered plane wave is unfocused on transmit, so this measures the
  // one-way receive beam, which for a Hann-apodized aperture is about 2*F#*lambda
  // — roughly twice the two-way width a focused transmit would give.
  check(w3 / lambda > 1.0f && w3 / lambda < 9.0f, "F/3 beam width is a few wavelengths");
  check_near(w3 / (w15 * 2.0f), 1.0, 0.25, "beam width scales with the f-number");
}

void test_doppler() {
  Section s("colour Doppler velocity estimation");
  Probe probe = standard_probe();
  Medium med;
  Acquisition acq = standard_acq(2200);
  acq.prf = 6000.0f;
  SimulatorConfig cfg;
  cfg.noise_db = -200.0f;
  Simulator sim(probe, med, acq, cfg);

  // A block of scatterers moving purely axially at a known speed. Purely axial
  // motion is the case where the estimator has no angle ambiguity, so the
  // answer is exactly recoverable.
  const real v_true = 0.15f;  // m/s, away from the probe (+z)
  Phantom ph;
  {
    Phantom block = Phantom::speckle_box(-0.004f, 0.004f, 0.028f, 0.032f, 1e-4f, 3000, 5);
    for (Scatterer& s : block.scatterers) s.velocity = Vec3(0, 0, v_true);
    ph.scatterers = block.scatterers;
  }

  const int n_ensemble = 16;
  const std::vector<Transmit> one = sim.make_plane_waves({0.0f});
  const std::vector<Transmit> seq = Simulator::repeat(one, n_ensemble);
  const ChannelData iq = demodulate(sim.simulate(seq, ph), {});

  const ScanGrid grid = ScanGrid::cartesian(-0.003f, 0.003f, 31, 0.029f, 0.031f, 61);
  Beamformer bf;
  const std::vector<Frame> ens = bf.beamform_sequence(iq, grid, 1);
  check(static_cast<int>(ens.size()) == n_ensemble, "ensemble has the expected number of frames");

  DopplerConfig dc;
  dc.wall_filter = WallFilter::None;  // nothing stationary to reject here
  dc.kernel_axial = 4;
  dc.kernel_lateral = 1;
  const ColorFlowMap cf = color_flow(ens, acq.prf, med.speed_of_sound, dc);

  double sum = 0;
  int n = 0;
  for (std::size_t k = 0; k < cf.velocity.size(); ++k)
    if (cf.valid[k]) { sum += cf.velocity[k]; n += 1; }
  check(n > 0, "some pixels produce a valid velocity");
  const double v_est = n ? sum / n : 0.0;
  std::printf("  (v_true = %.4f m/s, v_est = %.4f m/s, v_nyquist = %.3f m/s)\n", v_true, v_est,
              cf.nyquist_velocity);
  check_near(v_est, v_true, 0.02, "estimated axial velocity matches the truth");
  check(cf.nyquist_velocity > v_true, "the test velocity is below the aliasing limit");

  // The wall filter must suppress stationary tissue. Add a static speckle block
  // over the same region and confirm the velocity estimate survives.
  Phantom mixed = ph;
  Phantom tissue = Phantom::speckle_box(-0.004f, 0.004f, 0.028f, 0.032f, 1e-4f, 3000, 77, 4.0f);
  mixed.scatterers.insert(mixed.scatterers.end(), tissue.scatterers.begin(),
                          tissue.scatterers.end());
  const ChannelData iq2 = demodulate(sim.simulate(seq, mixed), {});
  const std::vector<Frame> ens2 = bf.beamform_sequence(iq2, grid, 1);

  DopplerConfig dc_off = dc;
  dc_off.wall_filter = WallFilter::None;
  const ColorFlowMap cf_off = color_flow(ens2, acq.prf, med.speed_of_sound, dc_off);
  DopplerConfig dc_on = dc;
  dc_on.wall_filter = WallFilter::PolynomialRegression;
  dc_on.polynomial_order = 1;
  const ColorFlowMap cf_on = color_flow(ens2, acq.prf, med.speed_of_sound, dc_on);

  auto mean_valid = [](const ColorFlowMap& c) {
    double s = 0; int n = 0;
    for (std::size_t k = 0; k < c.velocity.size(); ++k)
      if (c.valid[k]) { s += c.velocity[k]; ++n; }
    return n ? s / n : 0.0;
  };
  const double v_off = mean_valid(cf_off);
  const double v_on = mean_valid(cf_on);
  std::printf("  (with 4x stronger static clutter: no filter -> %.4f m/s, "
              "polynomial filter -> %.4f m/s)\n", v_off, v_on);
  check(std::abs(v_on - v_true) < std::abs(v_off - v_true),
        "the wall filter improves the estimate in the presence of clutter");
}

void test_displacement_and_strain() {
  Section s("displacement and strain estimation");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  SimulatorConfig cfg;
  cfg.noise_db = -200.0f;
  cfg.enable_motion = false;
  Simulator sim(probe, med, acq, cfg);

  Phantom before = Phantom::speckle_box(-0.006f, 0.006f, 0.026f, 0.034f, 1e-4f, 6000, 11);
  // Displace every scatterer by a known, sub-wavelength amount.
  const real d_true = 20e-6f;  // 20 microns, about 1/15 of a wavelength
  Phantom after = before;
  for (Scatterer& sc : after.scatterers) sc.position.z += d_true;

  const std::vector<Transmit> txs = sim.make_plane_waves({0.0f});
  const ChannelData iq_a = demodulate(sim.simulate(txs, before), {});
  const ChannelData iq_b = demodulate(sim.simulate(txs, after), {});

  const ScanGrid grid = ScanGrid::cartesian(-0.004f, 0.004f, 41, 0.028f, 0.032f, 121);
  Beamformer bf;
  const Frame fa = bf.beamform(iq_a, grid);
  const Frame fb = bf.beamform(iq_b, grid);

  const DisplacementMap dm = estimate_axial_displacement(fa, fb, med.speed_of_sound, 8, 2);
  double sum = 0, corr = 0;
  int n = 0;
  for (std::size_t k = 0; k < dm.axial.size(); ++k) {
    if (dm.correlation[k] < 0.9f) continue;
    sum += dm.axial[k];
    corr += dm.correlation[k];
    ++n;
  }
  check(n > 100, "most pixels track well between the two frames");
  const double d_est = n ? sum / n : 0.0;
  std::printf("  (d_true = %.2f um, d_est = %.2f um, mean correlation %.3f over %d pixels)\n",
              d_true * 1e6, d_est * 1e6, n ? corr / n : 0.0, n);
  check_near(d_est * 1e6, d_true * 1e6, 2.0, "estimated displacement matches the truth (um)");

  // A translation larger than half a wavelength cannot be recovered from phase
  // alone — and 1% compression of a 4 cm region displaces its bottom by 400 um,
  // so this is the normal case, not an edge case. The integer-lag search is what
  // makes it work.
  {
    const real d_big = 400e-6f;  // ~1.3 wavelengths at 5 MHz
    Phantom far = before;
    for (Scatterer& sc : far.scatterers) sc.position.z += d_big;
    const ChannelData iq_far = demodulate(sim.simulate(txs, far), {});
    const Frame ff = bf.beamform(iq_far, grid);

    const DisplacementMap phase_only =
        estimate_axial_displacement(fa, ff, med.speed_of_sound, 8, 2, 0);
    const DisplacementMap with_lag =
        estimate_axial_displacement(fa, ff, med.speed_of_sound, 8, 2, 12);

    auto median_of = [](const DisplacementMap& m) {
      std::vector<real> v;
      for (std::size_t k = 0; k < m.axial.size(); ++k)
        if (m.correlation[k] > 0.8f) v.push_back(m.axial[k]);
      if (v.empty()) return 0.0f;
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };
    const real m_phase = median_of(phase_only);
    const real m_lag = median_of(with_lag);
    std::printf("  (400 um translation: phase-only -> %.1f um, with lag search -> %.1f um)\n",
                m_phase * 1e6, m_lag * 1e6);
    check(std::abs(m_lag - d_big) < 20e-6f,
          "the coarse lag search recovers a displacement larger than a wavelength");
    check(std::abs(m_phase - d_big) > 50e-6f,
          "phase alone cannot (this is what the coarse stage is for)");
  }

  // A uniform translation has zero strain everywhere.
  const std::vector<real> strain = axial_strain(dm, 15);
  double smean = 0;
  int sn = 0;
  for (int i = 0; i < grid.n_axis0; ++i)
    for (int j = 20; j < grid.n_axis1 - 20; ++j) {
      smean += strain[static_cast<std::size_t>(i) * grid.n_axis1 + j];
      ++sn;
    }
  check_near(sn ? smean / sn : 0.0, 0.0, 5e-3, "uniform translation produces zero strain");

  // A linear compression produces a constant, known strain.
  const real strain_true = -0.005f;  // 0.5% compression
  Phantom compressed = before;
  const real z_ref = 0.030f;
  for (Scatterer& sc : compressed.scatterers)
    sc.position.z += strain_true * (sc.position.z - z_ref);
  const ChannelData iq_c = demodulate(sim.simulate(txs, compressed), {});
  const Frame fc = bf.beamform(iq_c, grid);
  const DisplacementMap dm2 = estimate_axial_displacement(fa, fc, med.speed_of_sound, 8, 2);
  const std::vector<real> strain2 = axial_strain(dm2, 21);
  double s2 = 0;
  int n2 = 0;
  for (int i = 0; i < grid.n_axis0; ++i)
    for (int j = 25; j < grid.n_axis1 - 25; ++j) {
      const std::size_t k = static_cast<std::size_t>(i) * grid.n_axis1 + j;
      if (dm2.correlation[k] < 0.9f) continue;
      s2 += strain2[k];
      ++n2;
    }
  const double s_est = n2 ? s2 / n2 : 0.0;
  std::printf("  (strain_true = %.4f, strain_est = %.4f over %d pixels)\n", strain_true, s_est, n2);
  check_near(s_est, strain_true, 1.5e-3, "estimated axial strain matches the applied compression");
}

void test_contrast() {
  Section s("contrast on an anechoic lesion");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  SimulatorConfig cfg;
  cfg.noise_db = -60.0f;
  Simulator sim(probe, med, acq, cfg);

  Phantom ph = Phantom::speckle_box(-0.012f, 0.012f, 0.020f, 0.040f, 2e-4f, 20000, 3);
  const Vec3 cyst(0.0f, 0, 0.030f);
  const real radius = 0.003f;
  ph.scale_amplitude_in_circle(cyst, radius, 0.0f);

  const ChannelData iq = demodulate(
      sim.simulate(sim.make_plane_waves({-0.1f, -0.05f, 0.0f, 0.05f, 0.1f}), ph), {});
  const ScanGrid grid = ScanGrid::cartesian(-0.010f, 0.010f, 121, 0.024f, 0.036f, 181);
  const Frame f = Beamformer().beamform(iq, grid);
  const std::vector<real> env = envelope(f);

  // Contrast: the anechoic region must actually be darker than the background.
  double in_sum = 0, out_sum = 0;
  int in_n = 0, out_n = 0;
  for (int i = 0; i < grid.n_axis0; ++i)
    for (int j = 0; j < grid.n_axis1; ++j) {
      const std::size_t k = static_cast<std::size_t>(i) * grid.n_axis1 + j;
      const real dx = grid.axis0[static_cast<std::size_t>(i)] - cyst.x;
      const real dz = grid.axis1[static_cast<std::size_t>(j)] - cyst.z;
      const real r = std::sqrt(dx * dx + dz * dz);
      if (r < radius * 0.6f) { in_sum += env[k] * env[k]; ++in_n; }
      else if (r > radius * 1.5f && r < radius * 2.5f) { out_sum += env[k] * env[k]; ++out_n; }
    }
  const double in_rms = std::sqrt(in_n ? in_sum / in_n : 0.0);
  const double out_rms = std::sqrt(out_n ? out_sum / out_n : 0.0);
  const double contrast_db = 20.0 * std::log10((in_rms + 1e-30) / (out_rms + 1e-30));
  std::printf("  (lesion contrast = %.1f dB below background)\n", contrast_db);
  check(contrast_db < -6.0, "the anechoic lesion is clearly darker than the background");
}

void test_scan_conversion() {
  Section s("sector scan conversion");
  const ScanGrid sector = ScanGrid::sector(-0.5f, 0.5f, 64, 0.01f, 0.06f, 128, Vec3(0, 0, 0));
  check(sector.type == GridType::Sector, "sector grid type");
  // A point at (theta, r) must map to the Cartesian position r*(sin, cos).
  const Vec3& p = sector.points[static_cast<std::size_t>(10) * 128 + 20];
  const real th = sector.axis0[10], r = sector.axis1[20];
  check_near(p.x, r * std::sin(th), 1e-9, "sector point x");
  check_near(p.z, r * std::cos(th), 1e-9, "sector point z");

  // Scan-converting a constant image must give a constant raster inside the
  // sector, and the fill value outside it.
  std::vector<real> img(static_cast<std::size_t>(sector.n_pixels()), 1.0f);
  const ScanConverted sc = scan_convert(img, sector, 128, 128, -1.0f);
  int inside = 0, outside = 0;
  for (real v : sc.pixels) {
    if (v > 0.99f && v < 1.01f) ++inside;
    else if (v < -0.5f) ++outside;
  }
  check(inside > 1000, "scan conversion fills the sector");
  check(outside > 1000, "scan conversion leaves the corners empty");
}

void test_pipeline() {
  Section s("end-to-end pipeline");
  const Probe probe = standard_probe();
  const Medium med;
  const Acquisition acq = standard_acq();
  Simulator sim(probe, med, acq, {});
  Phantom ph = Phantom::speckle_box(-0.012f, 0.012f, 0.015f, 0.045f, 2e-4f, 8000, 21);
  ph.add({Vec3(0.0f, 0, 0.030f), 30.0f, Vec3(0, 0, 0)});

  const ScanGrid grid = ScanGrid::cartesian(-0.012f, 0.012f, 96, 0.010f, 0.050f, 256);
  PipelineConfig pc;
  pc.demod.decimation = 2;
  ImagingPipeline pipe(grid, pc);

  const ChannelData raw = sim.simulate(sim.make_plane_waves({-0.1f, 0.0f, 0.1f}), ph);
  const std::vector<real> img = pipe.process_display(raw);

  check(static_cast<int>(img.size()) == grid.n_pixels(), "display image has one value per pixel");
  double lo = 1e30, hi = -1e30;
  for (real v : img) { lo = std::min(lo, static_cast<double>(v)); hi = std::max(hi, static_cast<double>(v)); }
  check(lo >= 0.0 && hi <= 1.0, "display image is normalized to [0, 1]");
  check(hi > 0.9, "display image reaches full brightness");
  check(pipe.frame_count() == 1, "pipeline counts frames");
  check(pipe.stats().total_ms > 0, "pipeline reports timing");
  std::printf("  (%s)\n", [&] {
    char b[160];
    std::snprintf(b, sizeof(b), "demod %.1f ms, beamform %.1f ms, post %.1f ms",
                  pipe.stats().demodulate_ms, pipe.stats().beamform_ms,
                  pipe.stats().postprocess_ms);
    return std::string(b);
  }().c_str());
}

int main() {
  test_windows();
  test_transmit_geometry();
  test_directivity();
  test_demodulation();
  test_point_target_localization();
  test_rf_and_iq_paths_agree();
  test_combiners();
  test_resolution_scales_with_aperture();
  test_doppler();
  test_displacement_and_strain();
  test_contrast();
  test_scan_conversion();
  test_pipeline();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
