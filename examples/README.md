# Examples

Run from the repository root with the package on the path:

```bash
export PYTHONPATH=$PWD/python
python examples/01_first_image.py
```

| Script | What it shows |
|---|---|
| `01_first_image.py` | The shortest complete path: describe hardware → simulate → beamform → measure. |
| `02_focused_vs_plane_wave.py` | The frame-rate/resolution trade, quantified: one focused scan against one and eleven plane waves. |
| `03_streaming.py` | A live loop over the `RawSource` protocol, with per-stage timing. Nothing in it knows the data is simulated. |
| `04_quantitative.py` | Colour flow and strain elastography — what the complex frame is *for*. |

Each writes images into `out/`.

**A note on runtime.** The forward simulation is the slow part, not the imaging
pipeline: a speckle phantom has tens of thousands of scatterers and the simulator
sums every element's contribution at each one. Expect tens of seconds to a couple
of minutes per script on a laptop; the beamforming that follows takes
milliseconds. Set `SimulatorConfig.full_transmit_diffraction = False` for a large
speed-up when transmit sidelobe detail does not matter.
