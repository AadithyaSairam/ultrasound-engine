// core/tests/benchmark.cpp — Where the time actually goes.
//
// "Live" is a throughput claim, so it needs a number attached. This binary
// measures each pipeline stage separately on a realistic imaging setup, because
// the interesting result is not the total but the ratio: beamforming dominates,
// and it dominates by an amount that tells you exactly what a GPU port would
// buy you.
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "usx/pipeline.hpp"
#include "usx/simulator.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace usx;
using Clock = std::chrono::steady_clock;

namespace {

double time_ms(const std::function<void()>& fn, int reps) {
  const auto t = Clock::now();
  for (int i = 0; i < reps; ++i) fn();
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count() / reps;
}

void row(const char* label, double ms, double per_pixel_ns = -1) {
  if (per_pixel_ns >= 0)
    std::printf("  %-38s %8.2f ms   %7.2f fps   %6.1f ns/pixel\n", label, ms,
                ms > 0 ? 1000.0 / ms : 0.0, per_pixel_ns);
  else
    std::printf("  %-38s %8.2f ms   %7.2f fps\n", label, ms, ms > 0 ? 1000.0 / ms : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
  int reps = (argc > 1) ? std::atoi(argv[1]) : 3;

  Probe probe;
  probe.n_elements = 128;
  probe.pitch = 300e-6f;
  probe.element_width = 270e-6f;
  probe.center_frequency = 5.0e6f;
  probe.bandwidth = 0.6f;

  Medium med;
  Acquisition acq;
  acq.sampling_frequency = 40.0e6f;
  acq.n_samples = 3072;
  acq.prf = 8000.0f;

  SimulatorConfig scfg;
  Simulator sim(probe, med, acq, scfg);

  Phantom ph = Phantom::speckle_box(-0.020f, 0.020f, 0.005f, 0.055f, 2e-4f, 20000, 3);
  ph.scale_amplitude_in_circle(Vec3(0.0f, 0, 0.030f), 0.004f, 0.0f);

  const std::vector<real> angles = {-0.16f, -0.096f, -0.032f, 0.032f, 0.096f, 0.16f};
  const std::vector<Transmit> txs = sim.make_plane_waves(angles);

  std::printf("usx benchmark\n");
  std::printf("  probe        %d elements @ %.1f MHz, pitch %.0f um\n", probe.n_elements,
              probe.center_frequency * 1e-6, probe.pitch * 1e6);
  std::printf("  acquisition  %d transmits x %d channels x %d samples @ %.0f MHz  (%.1f MB)\n",
              static_cast<int>(txs.size()), probe.n_elements, acq.n_samples,
              acq.sampling_frequency * 1e-6,
              static_cast<double>(txs.size()) * probe.n_elements * acq.n_samples * 4 / 1e6);
  std::printf("  phantom      %zu scatterers\n", ph.size());
#ifdef _OPENMP
  std::printf("  threads      %d\n", omp_get_max_threads());
#else
  std::printf("  threads      1 (built without OpenMP)\n");
#endif

  std::printf("\nforward simulation (development only — not part of the live path)\n");
  ChannelData raw;
  row("simulate 6 plane waves", time_ms([&] { raw = sim.simulate(txs, ph); }, 1));

  std::printf("\npipeline stages, 128 x 512 pixel image\n");
  const ScanGrid grid = ScanGrid::cartesian(-0.019f, 0.019f, 128, 0.005f, 0.055f, 512);
  const double n_pix = grid.n_pixels();

  ChannelData iq;
  DemodConfig dc;
  dc.decimation = 2;
  row("demodulate RF -> IQ (decimate 2x)", time_ms([&] { iq = demodulate(raw, dc); }, reps));

  BeamformerConfig bc;
  bc.f_number = 1.75f;
  Frame frame;

  bc.combiner = Combiner::DAS;
  double t = time_ms([&] { frame = Beamformer(bc).beamform(iq, grid); }, reps);
  row("beamform DAS, 6-angle compound", t, t * 1e6 / n_pix);

  bc.combiner = Combiner::DAS;
  bc.coherence_factor = true;
  t = time_ms([&] { frame = Beamformer(bc).beamform(iq, grid); }, reps);
  row("beamform DAS + coherence factor", t, t * 1e6 / n_pix);
  bc.coherence_factor = false;

  bc.combiner = Combiner::DMAS;
  t = time_ms([&] { frame = Beamformer(bc).beamform(iq, grid); }, reps);
  row("beamform DMAS, 6-angle compound", t, t * 1e6 / n_pix);

  bc.combiner = Combiner::MV;
  bc.mv_subarray = 16;
  t = time_ms([&] { frame = Beamformer(bc).beamform(iq, grid); }, 1);
  row("beamform MV (L=16), 6-angle compound", t, t * 1e6 / n_pix);

  bc.combiner = Combiner::DAS;
  bc.compounding = Compounding::Coherent;
  Beamformer bf1(bc);
  t = time_ms([&] { frame = bf1.beamform(iq, grid, 0, 1); }, reps);
  row("beamform DAS, single plane wave", t, t * 1e6 / n_pix);

  frame = Beamformer(BeamformerConfig{}).beamform(iq, grid);
  std::vector<real> env;
  row("envelope + log compress", time_ms([&] {
        env = envelope(frame);
        env = log_compress(env, 60.0f);
      }, reps));

  std::printf("\nend-to-end (RF in, display image out)\n");
  PipelineConfig pc;
  pc.demod.decimation = 2;
  ImagingPipeline pipe(grid, pc);
  std::vector<real> img;
  row("full pipeline, DAS 6-angle", time_ms([&] { img = pipe.process_display(raw); }, reps));

  pc.beamformer.compounding = Compounding::Coherent;
  ImagingPipeline pipe1(ScanGrid::cartesian(-0.019f, 0.019f, 96, 0.005f, 0.055f, 384), pc);
  row("full pipeline, 96 x 384 image", time_ms([&] { img = pipe1.process_display(raw); }, reps));

  std::printf("\nDoppler ensemble (16 packets x 1 plane wave, 64 x 256 pixels)\n");
  Acquisition dacq = acq;
  dacq.n_samples = 2048;
  Simulator dsim(probe, med, dacq, scfg);
  Phantom flow;
  flow.add_flow_tube(Vec3(0, 0, 0.028f), 0.0035f, 0.030f, 0.35f, 0.30f, 8000);
  const std::vector<Transmit> dtx = Simulator::repeat(dsim.make_plane_waves({0.0f}), 16);
  ChannelData draw = dsim.simulate(dtx, flow);
  ChannelData diq = demodulate(draw, dc);
  const ScanGrid dgrid = ScanGrid::cartesian(-0.014f, 0.014f, 64, 0.015f, 0.045f, 256);
  Beamformer dbf;
  std::vector<Frame> ens;
  row("beamform 16-frame ensemble",
      time_ms([&] { ens = dbf.beamform_sequence(diq, dgrid, 1); }, 1));
  ColorFlowMap cf;
  row("colour flow estimation",
      time_ms([&] { cf = color_flow(ens, dacq.prf, med.speed_of_sound, {}); }, reps));

  return 0;
}
