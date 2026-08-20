"""Where raw data comes from.

The engine does not care whether a frame of channel data was simulated,
loaded from a file, or streamed off a scanner. Every source implements the same
two-method protocol, so the beamformer, the metrics and the display are written
once and the simulator is just the source that happens to be available today.

    class RawSource:
        metadata -> (Probe, Medium, Acquisition, list[Transmit])
        frames() -> iterator of ChannelData

That is the whole interface. Adding hardware later means writing one class, not
touching the pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._usx import (Acquisition, ChannelData, Medium, Phantom, Probe, Simulator,
                   SimulatorConfig, Transmit)


@dataclass
class SourceMetadata:
    probe: Probe
    medium: Medium
    acquisition: Acquisition
    transmits: list = field(default_factory=list)


class RawSource(abc.ABC):
    """A source of raw pre-beamforming channel data."""

    @property
    @abc.abstractmethod
    def metadata(self) -> SourceMetadata:
        ...

    @abc.abstractmethod
    def frames(self):
        """Yield ChannelData objects, one per image frame."""

    def __iter__(self):
        return self.frames()

    @property
    def events_per_frame(self) -> int:
        return len(self.metadata.transmits)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SimulatedSource(RawSource):
    """Raw data synthesised on demand from a phantom.

    Set `n_frames` for a finite acquisition or None to stream forever. When the
    phantom contains moving scatterers, successive frames advance in time by the
    ensemble duration, so a live viewer fed by this source shows real motion.
    """

    def __init__(self, probe: Probe, medium: Medium, acquisition: Acquisition,
                 phantom: Phantom, transmits: list, config: SimulatorConfig | None = None,
                 n_frames: int | None = 1):
        self._sim = Simulator(probe, medium, acquisition, config or SimulatorConfig())
        self._meta = SourceMetadata(probe, medium, acquisition, list(transmits))
        self._phantom = phantom
        self._n_frames = n_frames

    @property
    def metadata(self) -> SourceMetadata:
        return self._meta

    @property
    def simulator(self) -> Simulator:
        return self._sim

    def frames(self):
        i = 0
        phantom = self._phantom
        dt = len(self._meta.transmits) / self._meta.acquisition.prf
        vel = np.asarray(phantom.velocities, dtype=np.float64)
        moving = bool(np.any(vel != 0.0))
        pos0 = np.asarray(phantom.positions, dtype=np.float64) if moving else None

        while self._n_frames is None or i < self._n_frames:
            if moving and i > 0:
                # Advance the phantom by one frame period. The simulator already
                # moves scatterers *within* an ensemble; this carries that motion
                # across frame boundaries so a stream is continuous rather than
                # resetting to t=0 every frame.
                p = phantom.copy()
                p.positions = (pos0 + vel * (i * dt)).astype(np.float32)
                yield self._sim.simulate(self._meta.transmits, p)
            else:
                yield self._sim.simulate(self._meta.transmits, phantom)
            i += 1


class FileSource(RawSource):
    """Raw data replayed from an .npz archive written by :func:`save_npz`.

    Deliberately a plain, self-describing format rather than a scanner vendor's:
    it keeps the geometry next to the samples, so a recording made today is still
    interpretable when the code has moved on.
    """

    def __init__(self, path, loop: bool = False):
        self.path = Path(path)
        self._loop = loop
        with np.load(self.path, allow_pickle=False) as z:
            self._rf = np.asarray(z["rf"], dtype=np.float32)
            probe = Probe()
            probe.n_elements = int(z["probe_n_elements"])
            probe.pitch = float(z["probe_pitch"])
            probe.element_width = float(z["probe_element_width"])
            probe.elevation_height = float(z["probe_elevation_height"])
            probe.center_frequency = float(z["probe_center_frequency"])
            probe.bandwidth = float(z["probe_bandwidth"])

            medium = Medium()
            medium.speed_of_sound = float(z["medium_speed_of_sound"])
            medium.attenuation = float(z["medium_attenuation"])

            acq = Acquisition()
            acq.sampling_frequency = float(z["acq_sampling_frequency"])
            acq.n_samples = int(z["acq_n_samples"])
            acq.t0 = float(z["acq_t0"])
            acq.prf = float(z["acq_prf"])

            transmits = _transmits_from_arrays(
                z["tx_type"], z["tx_angle"], z["tx_focus"], z["tx_origin"],
                z["tx_delays"], z["tx_apodization"])

        self._meta = SourceMetadata(probe, medium, acq, transmits)

    @property
    def metadata(self) -> SourceMetadata:
        return self._meta

    def frames(self):
        n_ev = len(self._meta.transmits)
        total = self._rf.shape[0] // n_ev if self._rf.ndim == 3 else 1
        while True:
            for f in range(max(total, 1)):
                block = self._rf[f * n_ev:(f + 1) * n_ev]
                yield ChannelData.from_rf(block, self._meta.probe, self._meta.medium,
                                          self._meta.acquisition, self._meta.transmits)
            if not self._loop:
                return


class ArraySource(RawSource):
    """Wrap an in-memory RF array — the adapter to write against when bringing
    up a real scanner. Feed it (events, channels, samples) blocks and everything
    downstream works unchanged."""

    def __init__(self, arrays, probe: Probe, medium: Medium, acquisition: Acquisition,
                 transmits: list):
        self._arrays = arrays
        self._meta = SourceMetadata(probe, medium, acquisition, list(transmits))

    @property
    def metadata(self) -> SourceMetadata:
        return self._meta

    def frames(self):
        for a in self._arrays:
            yield ChannelData.from_rf(np.asarray(a, dtype=np.float32), self._meta.probe,
                                      self._meta.medium, self._meta.acquisition,
                                      self._meta.transmits)


def _transmits_from_arrays(types, angles, focus, origin, delays, apod) -> list:
    from ._usx import TransmitType, Vec3
    out = []
    order = [TransmitType.FOCUSED, TransmitType.PLANE_WAVE, TransmitType.DIVERGING]
    for i in range(len(types)):
        t = Transmit()
        t.type = order[int(types[i])]
        t.angle = float(angles[i])
        t.focus = Vec3(*[float(v) for v in focus[i]])
        t.origin = Vec3(*[float(v) for v in origin[i]])
        t.delays = [float(v) for v in delays[i]]
        t.apodization = [float(v) for v in apod[i]]
        out.append(t)
    return out


def save_npz(path, data: ChannelData) -> Path:
    """Write channel data plus every parameter needed to interpret it.

    Saving the samples without the geometry is the classic way to render a
    dataset useless six months later: the delays cannot be recomputed and the
    depth scale cannot be recovered.
    """
    path = Path(path)
    txs = data.transmits
    type_index = {"FOCUSED": 0, "PLANE_WAVE": 1, "DIVERGING": 2}
    np.savez_compressed(
        path,
        rf=np.asarray(data.rf, dtype=np.float32),
        probe_n_elements=data.probe.n_elements,
        probe_pitch=data.probe.pitch,
        probe_element_width=data.probe.element_width,
        probe_elevation_height=data.probe.elevation_height,
        probe_center_frequency=data.probe.center_frequency,
        probe_bandwidth=data.probe.bandwidth,
        medium_speed_of_sound=data.medium.speed_of_sound,
        medium_attenuation=data.medium.attenuation,
        acq_sampling_frequency=data.acq.sampling_frequency,
        acq_n_samples=data.acq.n_samples,
        acq_t0=data.acq.t0,
        acq_prf=data.acq.prf,
        tx_type=np.array([type_index[str(t.type).split(".")[-1]] for t in txs], dtype=np.int32),
        tx_angle=np.array([t.angle for t in txs], dtype=np.float32),
        tx_focus=np.array([[t.focus.x, t.focus.y, t.focus.z] for t in txs], dtype=np.float32),
        tx_origin=np.array([[t.origin.x, t.origin.y, t.origin.z] for t in txs], dtype=np.float32),
        tx_delays=np.array([t.delays for t in txs], dtype=np.float32),
        tx_apodization=np.array([t.apodization for t in txs], dtype=np.float32),
    )
    return path
