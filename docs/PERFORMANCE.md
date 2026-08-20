# Performance

## Measured on the development machine

Everything below is `./build/usx_bench` on a **2-core x86-64 container**, so
these are pessimistic absolute numbers — the point is the *ratios*, which do not
change with core count.

Configuration: 128-element 5 MHz linear array, 6 steered plane waves,
128 channels × 3072 samples at 40 MHz (9.4 MB per acquisition), reconstructed to
a 128 × 512 pixel image.

| Stage | Time | ns/pixel |
|---|---:|---:|
| Demodulate RF → IQ (2× decimation) | 18.3 ms | — |
| **Beamform DAS**, 6-angle compound | **163 ms** | 2494 |
| Beamform DAS + coherence factor | 160 ms | 2439 |
| Beamform DMAS, 6-angle compound | 271 ms | 4136 |
| Beamform MV (L = 16), 6-angle compound | 1440 ms | 21977 |
| Beamform DAS, single plane wave | 72.5 ms | 1107 |
| Envelope + log compress | 1.1 ms | — |
| **End-to-end (RF in, display out)** | **174 ms** | 5.8 fps |
| End-to-end, 96 × 384 image | 101 ms | 9.9 fps |
| 16-frame Doppler ensemble (64 × 256) | 315 ms | — |
| Colour flow estimation | 30 ms | — |

Two conclusions:

**Beamforming is 94 % of the frame.** Everything else — demodulation, envelope,
log compression, scan conversion — is rounding error by comparison. Optimising
anything but the beamformer is wasted effort.

**Cost scales with transmits × channels × pixels, not with data size.** Six
angles cost 2.3× one angle, not 6×, because the per-pixel receive geometry
(apodization, delays, the sincos for the phase rotation) is computed once and
reused across every transmit. That hoist is worth more than it sounds.

## What was optimised, and by how much

These are the three changes that mattered, all found by measurement rather than
by guessing.

### 1. Hoisting the transmit aperture extent — the largest single win

`transmit_sensitivity` decides how strongly a transmit illuminates a pixel, and
its first step is finding the lateral extent of the firing aperture. Written
naively that is an O(n_elements) scan **inside a loop that already runs once per
pixel per transmit** — 128 × 128 × 512 × 6 ≈ 5 × 10⁷ wasted iterations per frame.
Computing it once per event moved it off the profile entirely.

### 2. Blocking the beamformer over depth — 244 ms → 163 ms (33 %)

The obvious loop order is *pixel outer, channel inner*. It is also the wrong one:
for a fixed pixel, the 128 channels live in 128 widely separated rows of a 9 MB
buffer, so every channel is a cache miss and the beamformer runs at memory speed.

The engine instead processes each image line in blocks of 64 depth samples and
puts the channel loop *outside* the depth loop. Within a block, the inner loop
walks one channel's trace in increasing depth — very nearly a sequential read —
and accumulates into a scratch buffer of 64 × 128 complex values, small enough to
stay in cache. The combiner then gathers from that hot buffer.

The block is padded by the minimum-variance range-averaging halo, so covariance
estimates at block boundaries are identical to those in the interior. Blocking
that changes the answer is not an optimisation.

### 3. Tabulating the demodulator's mixer — 61 ms → 18 ms (3.3×)

The mixing phase `−2πf·t` depends only on the sample index, never on which
channel or event the sample came from. Recomputing `sin`/`cos` per sample costs
`n_events × n_channels × n_samples` transcendental calls — 2.4 million for this
acquisition. Tabulating the rotator once and parallelising over traces gave most
of the 3.3×.

### What was deliberately *not* done

`-ffast-math` is off. The beamformer's phase arithmetic and the Cholesky solve in
the minimum-variance combiner both depend on IEEE semantics, and silent
reassociation there produces *wrong images* rather than slow ones — the kind of
wrong that still looks like an ultrasound image.

## Where the remaining time goes

For DAS, the inner loop per (pixel, channel, transmit) is:

1. two `sincos` calls for the phase rotation — hoisted so each is computed once
   per (pixel, channel) and once per (pixel, transmit), not per triple;
2. one gather + linear interpolation from the channel trace;
3. one complex multiply-accumulate.

At 2494 ns/pixel over 6 transmits × 128 channels = 768 channel-operations, that
is **3.2 ns per channel-operation on two cores**, or roughly 6.5 ns of
single-core work. That is memory-latency-dominated, not arithmetic-dominated: the
interpolated gather is a dependent load into a buffer far larger than L2.

## Scaling

The beamformer parallelises over image lines with `schedule(dynamic, 1)` and has
no cross-line dependencies, so it scales close to linearly with cores. On 2 cores
the observed speedup over a single-threaded build is about 1.9×. Extrapolating
the 128 × 512, 6-angle DAS figure:

| Cores | Estimated end-to-end | fps |
|---|---:|---:|
| 2 (measured) | 174 ms | 5.8 |
| 8 | ~48 ms | ~21 |
| 16 | ~26 ms | ~38 |

A single plane wave at 96 × 384 already runs at roughly 25 fps on 2 cores, which
is a usable live rate today for a low-resolution preview.

## What a GPU port would buy

Delay-and-sum is close to the ideal GPU workload: every pixel is independent,
every channel contribution is independent, the arithmetic is fused
multiply-accumulate, and the access pattern is a texture gather — which is
literally what texture units are for, including the linear interpolation for
free in hardware.

The realistic expectation is **two orders of magnitude**, putting a 6-angle
128 × 512 frame in the 1–2 ms range and making the *acquisition* the frame-rate
limit rather than the processing. Published CUDA plane-wave beamformers hit
several hundred to a thousand frames per second at this image size.

The parts that would *not* port trivially:

- **Minimum variance** needs a small Hermitian solve per pixel. It maps to a GPU
  (one warp per pixel, `L ≤ 32`), but it is a different kernel with different
  occupancy characteristics, not a recompile.
- **DMAS** is fine — the O(N) factorisation used here (see
  [THEORY.md](THEORY.md#delay-multiply-and-sum)) is a single pass over the
  aperture, exactly like DAS.
- **Demodulation** is a trivially parallel FIR and belongs on the GPU too, mostly
  to avoid a PCIe round trip rather than for its own cost.
- **Host↔device transfer** would become the real constraint: 9.4 MB per
  acquisition at 60 fps is 560 MB/s, which is fine over PCIe but means the raw
  data should be uploaded once and everything after it kept resident, rather than
  bouncing per stage.

## The simulator is not in the live path

`simulate 6 plane waves` takes about 12 s for 20,000 scatterers, and about 60 s
for the 80,000-scatterer phantoms the CLI demos use. That is fine: the simulator
is a development harness, and it is doing genuinely more work than the
beamformer — it sums every element's contribution at every scatterer to produce a
diffracted transmit field, rather than assuming an idealized wavefront (which is
[the whole point](THEORY.md#why-the-simulator-does-not-reuse-the-beamformers-model)).

`SimulatorConfig::full_transmit_diffraction = false` collapses the transmit field
to a single geometric arrival per scatterer and is much faster, at the cost of
transmit sidelobes. `incident_trim_db` is the finer dial on the same trade.

## Reproducing these numbers

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSX_NATIVE_ARCH=ON
cmake --build build -j
./build/usx_bench 5          # 5 repetitions per stage

PYTHONPATH=$PWD/python python -m usx bench --angles 7 --nx 192 --nz 512
```
