# Architecture

## The shape of the thing

```
                 ┌──────────────────────────────────────────────┐
   RawSource ───►│  ChannelData   (events × channels × samples)  │
  (sim / file /  │  + Probe + Medium + Acquisition + Transmit[]  │
   scanner)      └───────────────────┬──────────────────────────┘
                                     │
                          demodulate │  RF ──► complex baseband IQ
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  ChannelData (IQ)                            │
                 └───────────────────┬──────────────────────────┘
                                     │
                           beamform  │  delay · apodize · combine
                                     │  over a ScanGrid
                                     ▼
                 ┌──────────────────────────────────────────────┐
                 │  Frame  —  complex, on the scan grid          │◄── the product
                 └───┬──────────────────────────┬───────────────┘
                     │                          │
        display path │                          │ quantitative path
                     ▼                          ▼
        envelope → TGC → log        colour flow · power Doppler
        → scan convert → pixels     displacement → strain
```

The single most important structural decision is where that diagram forks. The
engine's output is `Frame`, which is **complex**. Log-compressed greyscale is a
consumer of it, not the thing itself. Everything on the right-hand branch —
Doppler, elastography, coherent compounding, speed-of-sound estimation — reads
phase, and phase is exactly what the display path throws away. Building an engine
whose output type is an image forecloses all of it.

## Files

| Path | What lives there |
|---|---|
| `core/include/usx/types.hpp` | The data model: `Probe`, `Medium`, `Acquisition`, `Transmit`, `ChannelData`, `ScanGrid`, `Frame`, apodization windows. |
| `core/include/usx/geometry.hpp` | Transmit delay laws, wavefront arrival times, element directivity, dynamic aperture. **The single source of truth for timing.** |
| `core/include/usx/simulator.hpp` | Point-scatterer forward model and the phantom builders. |
| `core/include/usx/demodulate.hpp` | RF → IQ quadrature demodulation, decimation, FIR design. |
| `core/include/usx/beamformer.hpp` | Delay-and-sum, DMAS, minimum variance; compounding modes; transmit sensitivity. |
| `core/include/usx/postprocess.hpp` | Envelope, TGC, log compression, scan conversion, Hilbert transform. |
| `core/include/usx/doppler.hpp` | Wall filters, colour flow, power Doppler, displacement, strain. |
| `core/include/usx/pipeline.hpp` | The stateful streaming front end: buffers, timing, brightness reference. |
| `bindings/pymodule.cpp` | pybind11: zero-copy numpy views, GIL released on every call that works. |
| `python/usx/ingest.py` | `RawSource` protocol; simulated, file and array sources. |
| `python/usx/phantoms.py` | Standard test targets: wires, cysts, flow tubes, stiff inclusions. |
| `python/usx/presets.py` | Probe archetypes and acquisition sizing. |
| `python/usx/metrics.py` | CNR, gCNR, contrast, speckle SNR, PSF measurement. |
| `python/usx/render.py` | B-mode, colour flow overlay, comparison figures. |
| `python/usx/cli.py` | `python -m usx {bmode,compare,doppler,strain,bench}`. |

## The core types, and why they are shaped that way

**`ChannelData` carries its own metadata.** The probe geometry, medium, timing
and full transmit list travel with the samples. A beamformer that needs a
side-channel of parameters to interpret its input is a beamformer that will
eventually be handed mismatched ones, and the failure is silent: the image simply
focuses at the wrong depth. It is also what makes a stored acquisition still
interpretable a year later.

**`ScanGrid` is an explicit list of points, not a formula.** One beamformer
therefore serves Cartesian rasters (linear arrays), polar sectors (phased
arrays), a single scanline, or a scattered set of points for a spot measurement.
The alternative — hard-coding the raster into the beamformer — is what forces a
second beamformer the first time a sector format is needed.

**`Frame` is complex.** See above.

**`Transmit` carries its delay law, not just its parameters.** The delays are
computed once, stored, and used by both the simulator and (via
`transmit_arrival_time`) the beamformer, so the two cannot drift apart.

## Adding a data source

Implement two methods:

```python
class MyScanner(usx.ingest.RawSource):
    @property
    def metadata(self) -> usx.ingest.SourceMetadata:
        return usx.ingest.SourceMetadata(probe, medium, acquisition, transmits)

    def frames(self):
        while self.streaming:
            block = self.read_block()          # (events, channels, samples), float32
            yield usx.ChannelData.from_rf(block, *self.metadata_tuple)
```

Nothing downstream changes. `ArraySource` is the worked example, and it is the
class to copy when adapting a public dataset (PICMUS, CUBDL) or a scanner SDK.
The one thing that must be got right is the **`t0` convention** — see
[THEORY.md](THEORY.md#the-t--0-convention). If a recording's sample 0 corresponds
to the first element firing rather than to the wavefront crossing the array
origin, set `Acquisition.t0` to the offset that `compute_transmit_delays` returns.

## Adding a beamformer

A combiner is a pure function from the delay-compensated channel vector to one
complex pixel value. Add a case to `Combiner`, write the function next to
`combine_das` in `core/src/beamformer.cpp`, and dispatch to it. The delay,
apodization, compounding and blocking machinery is shared and does not need to be
touched — which is the point of separating the two halves.

## Threading and memory

- The simulator parallelises over scatterers, each thread accumulating into a
  private copy of the event's channel buffer, reduced at the end. Scatterers are
  independent, so this scales cleanly; atomics on every sample would be dominated
  by contention.
- The beamformer parallelises over image lines, and within a line processes depth
  in blocks of 64 pixels so that the working set stays in cache. See
  [PERFORMANCE.md](PERFORMANCE.md) for why that matters more than it sounds.
- The demodulator parallelises over traces and hoists the mixer's rotator table,
  which depends only on the sample index.
- Every pybind11 entry point that does real work releases the GIL, so a Python
  application can run acquisition, display and inference concurrently with
  beamforming.

## Testing strategy

`core/tests/test_main.cpp` asserts on **physics with known answers**, not on
absence of crashes:

- every element of a focused transmit reaches the focus at the same instant;
- a steered plane wave arrives at the projection of the point onto its normal;
- the delay law and the wavefront model agree at the array itself — this is what
  guarantees the simulator and the beamformer share a clock;
- point targets reconstruct within 0.2 mm of where they were placed, under three
  different transmit schemes;
- the RF and IQ beamforming paths find the same target;
- halving the receive f-number halves the beam width;
- DMAS and MV resolve finer than DAS;
- a scatterer moving at a known velocity produces that velocity;
- a phantom translated by 20 µm measures as 20 µm; compressed by 0.5 %, as 0.5 %;
- an anechoic lesion is measurably darker than its surroundings.

The Python suite covers the boundary: that numpy views really are views, that
the ingest sources are interchangeable, that a file round trip preserves both
samples and geometry, and that gCNR really is invariant to monotone transforms
while CNR is not.
