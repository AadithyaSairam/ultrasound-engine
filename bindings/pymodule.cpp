// bindings/pymodule.cpp — pybind11 bindings for the usx core.
//
// The design rule here is that numpy arrays view the C++ buffers rather than
// copying them. A single acquisition is tens of megabytes and a live pipeline
// touches it several times per frame; copying on every crossing of the language
// boundary would dominate the runtime and defeat the point of having a C++ core
// at all. Every array-returning accessor therefore hands back a view whose
// lifetime is tied to the owning object via pybind11's `base` mechanism.
//
// The other rule: every call that does real work releases the GIL, so a Python
// application can run acquisition, display and inference concurrently with
// beamforming.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "usx/beamformer.hpp"
#include "usx/demodulate.hpp"
#include "usx/doppler.hpp"
#include "usx/geometry.hpp"
#include "usx/pipeline.hpp"
#include "usx/postprocess.hpp"
#include "usx/simulator.hpp"
#include "usx/types.hpp"

namespace py = pybind11;
using namespace usx;

namespace {

// Wrap a raw buffer as a numpy array that keeps `owner` alive and does not copy.
template <typename T>
py::array_t<T> view(T* ptr, std::vector<py::ssize_t> shape, py::object owner) {
  std::vector<py::ssize_t> strides(shape.size());
  py::ssize_t s = static_cast<py::ssize_t>(sizeof(T));
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = s;
    s *= shape[i];
  }
  return py::array_t<T>(shape, strides, ptr, std::move(owner));
}

template <typename T>
py::array_t<T> copy_vector(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
  py::array_t<T> a(shape);
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

}  // namespace

PYBIND11_MODULE(_usx, m) {
  m.doc() = "usx — a from-scratch ultrasound imaging engine (C++ core)";

  // --- enums ---------------------------------------------------------------
  py::enum_<Window>(m, "Window")
      .value("RECT", Window::Rect)
      .value("HANN", Window::Hann)
      .value("HAMMING", Window::Hamming)
      .value("TUKEY", Window::Tukey)
      .value("BLACKMAN", Window::Blackman);

  py::enum_<TransmitType>(m, "TransmitType")
      .value("FOCUSED", TransmitType::Focused)
      .value("PLANE_WAVE", TransmitType::PlaneWave)
      .value("DIVERGING", TransmitType::Diverging);

  py::enum_<GridType>(m, "GridType")
      .value("CARTESIAN", GridType::Cartesian)
      .value("SECTOR", GridType::Sector);

  py::enum_<Combiner>(m, "Combiner")
      .value("DAS", Combiner::DAS)
      .value("DMAS", Combiner::DMAS)
      .value("MV", Combiner::MV);

  py::enum_<Compounding>(m, "Compounding")
      .value("COHERENT", Compounding::Coherent)
      .value("INCOHERENT", Compounding::Incoherent)
      .value("SELECT", Compounding::Select);

  py::enum_<WallFilter>(m, "WallFilter")
      .value("NONE", WallFilter::None)
      .value("POLYNOMIAL", WallFilter::PolynomialRegression)
      .value("MEAN", WallFilter::MeanSubtraction);

  // --- geometry ------------------------------------------------------------
  py::class_<Vec3>(m, "Vec3")
      .def(py::init<>())
      .def(py::init<real, real, real>(), py::arg("x"), py::arg("y"), py::arg("z"))
      .def_readwrite("x", &Vec3::x)
      .def_readwrite("y", &Vec3::y)
      .def_readwrite("z", &Vec3::z)
      .def("__repr__", [](const Vec3& v) {
        return "Vec3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
               std::to_string(v.z) + ")";
      });

  py::class_<Probe>(m, "Probe")
      .def(py::init<>())
      .def_readwrite("n_elements", &Probe::n_elements)
      .def_readwrite("pitch", &Probe::pitch)
      .def_readwrite("element_width", &Probe::element_width)
      .def_readwrite("elevation_height", &Probe::elevation_height)
      .def_readwrite("center_frequency", &Probe::center_frequency)
      .def_readwrite("bandwidth", &Probe::bandwidth)
      .def("element_position", &Probe::element_position)
      .def("aperture_width", &Probe::aperture_width)
      .def("wavelength", &Probe::wavelength, py::arg("speed_of_sound") = 1540.0f)
      .def_property_readonly("element_x",
                             [](const Probe& p) {
                               std::vector<real> x(static_cast<std::size_t>(p.n_elements));
                               for (int i = 0; i < p.n_elements; ++i)
                                 x[static_cast<std::size_t>(i)] = p.element_position(i).x;
                               return copy_vector(x, {p.n_elements});
                             })
      .def("validate", &Probe::validate);

  py::class_<Medium>(m, "Medium")
      .def(py::init<>())
      .def_readwrite("speed_of_sound", &Medium::speed_of_sound)
      .def_readwrite("attenuation", &Medium::attenuation)
      .def_readwrite("density", &Medium::density);

  py::class_<Acquisition>(m, "Acquisition")
      .def(py::init<>())
      .def_readwrite("sampling_frequency", &Acquisition::sampling_frequency)
      .def_readwrite("n_samples", &Acquisition::n_samples)
      .def_readwrite("t0", &Acquisition::t0)
      .def_readwrite("prf", &Acquisition::prf)
      .def("max_depth", &Acquisition::max_depth, py::arg("speed_of_sound") = 1540.0f);

  py::class_<Transmit>(m, "Transmit")
      .def(py::init<>())
      .def_readwrite("type", &Transmit::type)
      .def_readwrite("angle", &Transmit::angle)
      .def_readwrite("focus", &Transmit::focus)
      .def_readwrite("origin", &Transmit::origin)
      .def_readwrite("delays", &Transmit::delays)
      .def_readwrite("apodization", &Transmit::apodization)
      .def_property_readonly("n_active", &Transmit::n_active);

  m.def(
      "compute_transmit_delays",
      [](const Probe& probe, const Medium& medium, Transmit& tx, Window w, real f_number) {
        real t0 = 0;
        compute_transmit_delays(probe, medium, tx, w, f_number, &t0);
        return t0;
      },
      py::arg("probe"), py::arg("medium"), py::arg("transmit"),
      py::arg("window") = Window::Hann, py::arg("f_number") = 2.0f,
      "Fill in a transmit's delay law and apodization; returns the hardware t0 offset.");

  m.def("transmit_arrival_time", &transmit_arrival_time, py::arg("transmit"), py::arg("point"),
        py::arg("speed_of_sound"));
  m.def("transmit_sensitivity", &transmit_sensitivity, py::arg("probe"), py::arg("transmit"),
        py::arg("point"), py::arg("wavelength"));
  m.def("element_directivity", &element_directivity, py::arg("element_width"),
        py::arg("wavelength"), py::arg("sin_theta"));

  // --- scan grid -----------------------------------------------------------
  py::class_<ScanGrid>(m, "ScanGrid")
      .def(py::init<>())
      .def_static("cartesian", &ScanGrid::cartesian, py::arg("x_min"), py::arg("x_max"),
                  py::arg("nx"), py::arg("z_min"), py::arg("z_max"), py::arg("nz"))
      .def_static("sector", &ScanGrid::sector, py::arg("theta_min"), py::arg("theta_max"),
                  py::arg("n_theta"), py::arg("r_min"), py::arg("r_max"), py::arg("n_r"),
                  py::arg("apex"))
      .def_readonly("type", &ScanGrid::type)
      .def_readonly("n_axis0", &ScanGrid::n_axis0)
      .def_readonly("n_axis1", &ScanGrid::n_axis1)
      .def_readonly("apex", &ScanGrid::apex)
      .def_property_readonly("axis0", [](ScanGrid& g) { return copy_vector(g.axis0, {g.n_axis0}); })
      .def_property_readonly("axis1", [](ScanGrid& g) { return copy_vector(g.axis1, {g.n_axis1}); })
      .def_property_readonly("shape",
                             [](const ScanGrid& g) { return py::make_tuple(g.n_axis0, g.n_axis1); })
      .def_property_readonly("n_pixels", &ScanGrid::n_pixels)
      .def_property_readonly("points",
                             [](ScanGrid& g) {
                               py::array_t<real> a({g.n_pixels(), 3});
                               auto r = a.mutable_unchecked<2>();
                               for (int k = 0; k < g.n_pixels(); ++k) {
                                 r(k, 0) = g.points[static_cast<std::size_t>(k)].x;
                                 r(k, 1) = g.points[static_cast<std::size_t>(k)].y;
                                 r(k, 2) = g.points[static_cast<std::size_t>(k)].z;
                               }
                               return a;
                             });

  // --- channel data --------------------------------------------------------
  py::class_<ChannelData>(m, "ChannelData")
      .def(py::init<>())
      .def_readwrite("probe", &ChannelData::probe)
      .def_readwrite("medium", &ChannelData::medium)
      .def_readwrite("acq", &ChannelData::acq)
      .def_readwrite("transmits", &ChannelData::transmits)
      .def_readonly("is_iq", &ChannelData::is_iq)
      .def_readonly("demodulation_frequency", &ChannelData::demodulation_frequency)
      .def_property_readonly("n_events", &ChannelData::n_events)
      .def_property_readonly("n_channels", &ChannelData::n_channels)
      .def_property_readonly("n_samples", &ChannelData::n_samples)
      .def_property_readonly("shape",
                             [](const ChannelData& d) {
                               return py::make_tuple(d.n_events(), d.n_channels(), d.n_samples());
                             })
      // Zero-copy views. Mutating these mutates the C++ buffer, which is the
      // point: a Python-side filter can operate in place on 50 MB of channel
      // data without a round trip.
      .def_property_readonly(
          "rf",
          [](py::object self) {
            ChannelData& d = self.cast<ChannelData&>();
            if (d.is_iq) throw std::runtime_error("ChannelData holds IQ, not RF");
            return view<real>(d.rf.data(), {d.n_events(), d.n_channels(), d.n_samples()}, self);
          })
      .def_property_readonly(
          "iq",
          [](py::object self) {
            ChannelData& d = self.cast<ChannelData&>();
            if (!d.is_iq) throw std::runtime_error("ChannelData holds RF, not IQ");
            return view<std::complex<real>>(d.iq.data(),
                                            {d.n_events(), d.n_channels(), d.n_samples()}, self);
          })
      .def("validate", &ChannelData::validate)
      .def_static(
          "from_rf",
          [](py::array_t<real, py::array::c_style | py::array::forcecast> arr, Probe probe,
             Medium medium, Acquisition acq, std::vector<Transmit> transmits) {
            if (arr.ndim() != 3)
              throw std::invalid_argument("from_rf expects an array of shape (events, channels, samples)");
            ChannelData d;
            d.probe = probe;
            d.medium = medium;
            d.acq = acq;
            d.acq.n_samples = static_cast<int>(arr.shape(2));
            d.transmits = std::move(transmits);
            d.is_iq = false;
            d.allocate();
            if (static_cast<std::size_t>(arr.size()) != d.size())
              throw std::invalid_argument("from_rf: array shape does not match probe/transmits");
            std::memcpy(d.rf.data(), arr.data(), d.size() * sizeof(real));
            return d;
          },
          py::arg("array"), py::arg("probe"), py::arg("medium"), py::arg("acquisition"),
          py::arg("transmits"),
          "Adopt an external RF array (events, channels, samples) as ChannelData.");

  // --- frame ---------------------------------------------------------------
  py::class_<Frame>(m, "Frame")
      .def(py::init<>())
      .def_readonly("grid", &Frame::grid)
      .def_readwrite("demodulation_frequency", &Frame::demodulation_frequency)
      .def_readwrite("timestamp", &Frame::timestamp)
      .def_readwrite("sequence", &Frame::sequence)
      .def_property_readonly("data",
                             [](py::object self) {
                               Frame& f = self.cast<Frame&>();
                               return view<std::complex<real>>(
                                   f.data.data(), {f.grid.n_axis0, f.grid.n_axis1}, self);
                             })
      .def_property_readonly("shape", [](const Frame& f) {
        return py::make_tuple(f.grid.n_axis0, f.grid.n_axis1);
      });

  // --- simulator -----------------------------------------------------------
  py::class_<Scatterer>(m, "Scatterer")
      .def(py::init<>())
      .def_readwrite("position", &Scatterer::position)
      .def_readwrite("amplitude", &Scatterer::amplitude)
      .def_readwrite("velocity", &Scatterer::velocity);

  py::class_<Phantom>(m, "Phantom")
      .def(py::init<>())
      .def("add", &Phantom::add)
      .def("__len__", &Phantom::size)
      .def_static("point_targets", &Phantom::point_targets, py::arg("positions"),
                  py::arg("amplitude") = 1.0f)
      .def_static("speckle_box", &Phantom::speckle_box, py::arg("x_min"), py::arg("x_max"),
                  py::arg("z_min"), py::arg("z_max"), py::arg("y_half_thickness"),
                  py::arg("n_scatterers"), py::arg("seed") = 1234u,
                  py::arg("amplitude_sigma") = 1.0f)
      .def_static("recommended_scatterer_count", &Phantom::recommended_scatterer_count,
                  py::arg("probe"), py::arg("medium"), py::arg("x_span"), py::arg("z_span"),
                  py::arg("y_span"), py::arg("f_number") = 2.0f, py::arg("per_cell") = 10.0f)
      .def("scale_amplitude_in_circle", &Phantom::scale_amplitude_in_circle, py::arg("center"),
           py::arg("radius"), py::arg("scale"))
      .def("add_flow_tube", &Phantom::add_flow_tube, py::arg("center"), py::arg("radius"),
           py::arg("length"), py::arg("axis_angle"), py::arg("peak_velocity"),
           py::arg("n_scatterers"), py::arg("seed") = 99u, py::arg("amplitude") = 0.05f)
      .def("extend", [](Phantom& self, const Phantom& other) {
        self.scatterers.insert(self.scatterers.end(), other.scatterers.begin(),
                               other.scatterers.end());
      })
      // Bulk numpy accessors. A speckle phantom has hundreds of thousands of
      // scatterers, so anything that touches them all — applying a deformation
      // field, masking a region — has to be vectorised rather than looped in
      // Python one object at a time.
      .def_property(
          "positions",
          [](const Phantom& p) {
            py::array_t<real> a({static_cast<py::ssize_t>(p.scatterers.size()), py::ssize_t(3)});
            auto r = a.mutable_unchecked<2>();
            for (std::size_t i = 0; i < p.scatterers.size(); ++i) {
              r(static_cast<py::ssize_t>(i), 0) = p.scatterers[i].position.x;
              r(static_cast<py::ssize_t>(i), 1) = p.scatterers[i].position.y;
              r(static_cast<py::ssize_t>(i), 2) = p.scatterers[i].position.z;
            }
            return a;
          },
          [](Phantom& p, py::array_t<real, py::array::c_style | py::array::forcecast> a) {
            if (a.ndim() != 2 || a.shape(1) != 3)
              throw std::invalid_argument("positions must have shape (n, 3)");
            p.scatterers.resize(static_cast<std::size_t>(a.shape(0)));
            auto r = a.unchecked<2>();
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
              p.scatterers[static_cast<std::size_t>(i)].position = Vec3(r(i, 0), r(i, 1), r(i, 2));
          })
      .def_property(
          "amplitudes",
          [](const Phantom& p) {
            py::array_t<real> a(static_cast<py::ssize_t>(p.scatterers.size()));
            auto r = a.mutable_unchecked<1>();
            for (std::size_t i = 0; i < p.scatterers.size(); ++i)
              r(static_cast<py::ssize_t>(i)) = p.scatterers[i].amplitude;
            return a;
          },
          [](Phantom& p, py::array_t<real, py::array::c_style | py::array::forcecast> a) {
            if (static_cast<std::size_t>(a.size()) != p.scatterers.size())
              throw std::invalid_argument("amplitudes length must match the scatterer count");
            auto r = a.unchecked<1>();
            for (py::ssize_t i = 0; i < a.size(); ++i)
              p.scatterers[static_cast<std::size_t>(i)].amplitude = r(i);
          })
      .def_property(
          "velocities",
          [](const Phantom& p) {
            py::array_t<real> a({static_cast<py::ssize_t>(p.scatterers.size()), py::ssize_t(3)});
            auto r = a.mutable_unchecked<2>();
            for (std::size_t i = 0; i < p.scatterers.size(); ++i) {
              r(static_cast<py::ssize_t>(i), 0) = p.scatterers[i].velocity.x;
              r(static_cast<py::ssize_t>(i), 1) = p.scatterers[i].velocity.y;
              r(static_cast<py::ssize_t>(i), 2) = p.scatterers[i].velocity.z;
            }
            return a;
          },
          [](Phantom& p, py::array_t<real, py::array::c_style | py::array::forcecast> a) {
            if (a.ndim() != 2 || a.shape(1) != 3 ||
                static_cast<std::size_t>(a.shape(0)) != p.scatterers.size())
              throw std::invalid_argument("velocities must have shape (n, 3) matching the phantom");
            auto r = a.unchecked<2>();
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
              p.scatterers[static_cast<std::size_t>(i)].velocity = Vec3(r(i, 0), r(i, 1), r(i, 2));
          })
      .def("copy", [](const Phantom& p) { return Phantom(p); });

  py::class_<SimulatorConfig>(m, "SimulatorConfig")
      .def(py::init<>())
      .def_readwrite("pulse_cycles", &SimulatorConfig::pulse_cycles)
      .def_readwrite("tx_window", &SimulatorConfig::tx_window)
      .def_readwrite("tx_f_number", &SimulatorConfig::tx_f_number)
      .def_readwrite("oversampling", &SimulatorConfig::oversampling)
      .def_readwrite("full_transmit_diffraction", &SimulatorConfig::full_transmit_diffraction)
      .def_readwrite("incident_trim_db", &SimulatorConfig::incident_trim_db)
      .def_readwrite("max_incident_window_us", &SimulatorConfig::max_incident_window_us)
      .def_readwrite("apply_attenuation", &SimulatorConfig::apply_attenuation)
      .def_readwrite("apply_directivity", &SimulatorConfig::apply_directivity)
      .def_readwrite("apply_spreading", &SimulatorConfig::apply_spreading)
      .def_readwrite("noise_db", &SimulatorConfig::noise_db)
      .def_readwrite("noise_seed", &SimulatorConfig::noise_seed)
      .def_readwrite("enable_motion", &SimulatorConfig::enable_motion);

  py::class_<Simulator>(m, "Simulator")
      .def(py::init<Probe, Medium, Acquisition, SimulatorConfig>(), py::arg("probe"),
           py::arg("medium"), py::arg("acquisition"), py::arg("config") = SimulatorConfig{})
      .def("make_plane_waves", &Simulator::make_plane_waves, py::arg("angles"))
      .def("make_focused_scan", &Simulator::make_focused_scan, py::arg("n_lines"),
           py::arg("x_min"), py::arg("x_max"), py::arg("focal_depth"), py::arg("angle") = 0.0f)
      .def("make_diverging_waves", &Simulator::make_diverging_waves, py::arg("angles"),
           py::arg("virtual_source_depth"))
      .def_static("repeat", &Simulator::repeat, py::arg("transmits"), py::arg("n_ensemble"))
      .def("simulate", &Simulator::simulate, py::arg("transmits"), py::arg("phantom"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("probe", py::overload_cast<>(&Simulator::probe, py::const_))
      .def_property_readonly("medium", py::overload_cast<>(&Simulator::medium, py::const_))
      .def_property_readonly("acquisition",
                             py::overload_cast<>(&Simulator::acquisition, py::const_))
      .def("pulse", [](const Simulator& s) {
        std::vector<real> p = s.pulse(nullptr);
        return copy_vector(p, {static_cast<py::ssize_t>(p.size())});
      });

  // --- demodulation --------------------------------------------------------
  py::class_<DemodConfig>(m, "DemodConfig")
      .def(py::init<>())
      .def_readwrite("demodulation_frequency", &DemodConfig::demodulation_frequency)
      .def_readwrite("cutoff_fraction", &DemodConfig::cutoff_fraction)
      .def_readwrite("filter_taps", &DemodConfig::filter_taps)
      .def_readwrite("decimation", &DemodConfig::decimation);

  m.def("demodulate", &demodulate, py::arg("channel_data"), py::arg("config") = DemodConfig{},
        py::call_guard<py::gil_scoped_release>());
  m.def("design_lowpass", [](int n, real fc) {
    std::vector<real> h = design_lowpass(n, fc);
    return copy_vector(h, {static_cast<py::ssize_t>(h.size())});
  }, py::arg("n_taps"), py::arg("normalized_cutoff"));

  // --- beamformer ----------------------------------------------------------
  py::class_<BeamformerConfig>(m, "BeamformerConfig")
      .def(py::init<>())
      .def_readwrite("f_number", &BeamformerConfig::f_number)
      .def_readwrite("rx_window", &BeamformerConfig::rx_window)
      .def_readwrite("tukey_alpha", &BeamformerConfig::tukey_alpha)
      .def_readwrite("combiner", &BeamformerConfig::combiner)
      .def_readwrite("compounding", &BeamformerConfig::compounding)
      .def_readwrite("transmit_weighting", &BeamformerConfig::transmit_weighting)
      .def_readwrite("coherence_factor", &BeamformerConfig::coherence_factor)
      .def_readwrite("mv_subarray", &BeamformerConfig::mv_subarray)
      .def_readwrite("mv_diagonal_loading", &BeamformerConfig::mv_diagonal_loading)
      .def_readwrite("mv_range_average", &BeamformerConfig::mv_range_average);

  py::class_<Beamformer>(m, "Beamformer")
      .def(py::init<BeamformerConfig>(), py::arg("config") = BeamformerConfig{})
      .def("beamform", &Beamformer::beamform, py::arg("channel_data"), py::arg("grid"),
           py::arg("event_begin") = 0, py::arg("event_end") = -1,
           py::call_guard<py::gil_scoped_release>())
      .def("beamform_sequence", &Beamformer::beamform_sequence, py::arg("channel_data"),
           py::arg("grid"), py::arg("events_per_frame"),
           py::call_guard<py::gil_scoped_release>())
      .def_property("config", py::overload_cast<>(&Beamformer::config, py::const_),
                    [](Beamformer& b, const BeamformerConfig& c) { b.config() = c; });

  // --- post-processing -----------------------------------------------------
  m.def("envelope", [](const Frame& f) {
    std::vector<real> e = envelope(f);
    return copy_vector(e, {f.grid.n_axis0, f.grid.n_axis1});
  }, py::arg("frame"));

  m.def("log_compress",
        [](py::array_t<real, py::array::c_style | py::array::forcecast> env, real dr, real gain,
           real reference) {
          std::vector<real> v(env.data(), env.data() + env.size());
          std::vector<real> out = log_compress(v, dr, gain, reference);
          py::array_t<real> a(std::vector<py::ssize_t>(env.shape(), env.shape() + env.ndim()));
          std::memcpy(a.mutable_data(), out.data(), out.size() * sizeof(real));
          return a;
        },
        py::arg("envelope"), py::arg("dynamic_range_db") = 60.0f, py::arg("gain_db") = 0.0f,
        py::arg("reference") = -1.0f);

  m.def("apply_tgc",
        [](py::array_t<real, py::array::c_style | py::array::forcecast> env, const ScanGrid& grid,
           real atten, real f0) {
          std::vector<real> v(env.data(), env.data() + env.size());
          std::vector<real> out = apply_tgc(v, grid, atten, f0);
          return copy_vector(out, {grid.n_axis0, grid.n_axis1});
        },
        py::arg("envelope"), py::arg("grid"), py::arg("attenuation_db_cm_mhz"),
        py::arg("center_frequency_hz"));

  py::class_<ScanConverted>(m, "ScanConverted")
      .def_readonly("width", &ScanConverted::width)
      .def_readonly("height", &ScanConverted::height)
      .def_readonly("x_min", &ScanConverted::x_min)
      .def_readonly("x_max", &ScanConverted::x_max)
      .def_readonly("z_min", &ScanConverted::z_min)
      .def_readonly("z_max", &ScanConverted::z_max)
      .def_property_readonly("extent",
                            [](const ScanConverted& s) {
                              return py::make_tuple(s.x_min, s.x_max, s.z_max, s.z_min);
                            })
      .def_property_readonly("pixels", [](py::object self) {
        ScanConverted& s = self.cast<ScanConverted&>();
        return view<real>(s.pixels.data(), {s.height, s.width}, self);
      });

  m.def("scan_convert",
        [](py::array_t<real, py::array::c_style | py::array::forcecast> img, const ScanGrid& grid,
           int width, int height, real fill) {
          std::vector<real> v(img.data(), img.data() + img.size());
          return scan_convert(v, grid, width, height, fill);
        },
        py::arg("image"), py::arg("grid"), py::arg("width"), py::arg("height"),
        py::arg("fill_value") = 0.0f, py::call_guard<py::gil_scoped_release>());

  // --- doppler / elastography ---------------------------------------------
  py::class_<DopplerConfig>(m, "DopplerConfig")
      .def(py::init<>())
      .def_readwrite("wall_filter", &DopplerConfig::wall_filter)
      .def_readwrite("polynomial_order", &DopplerConfig::polynomial_order)
      .def_readwrite("loupas", &DopplerConfig::loupas)
      .def_readwrite("kernel_axial", &DopplerConfig::kernel_axial)
      .def_readwrite("kernel_lateral", &DopplerConfig::kernel_lateral)
      .def_readwrite("power_threshold_db", &DopplerConfig::power_threshold_db)
      .def_readwrite("coherence_threshold", &DopplerConfig::coherence_threshold);

  py::class_<ColorFlowMap>(m, "ColorFlowMap")
      .def_readonly("grid", &ColorFlowMap::grid)
      .def_readonly("nyquist_velocity", &ColorFlowMap::nyquist_velocity)
      .def_property_readonly("velocity",
                             [](py::object self) {
                               ColorFlowMap& c = self.cast<ColorFlowMap&>();
                               return view<real>(c.velocity.data(),
                                                 {c.grid.n_axis0, c.grid.n_axis1}, self);
                             })
      .def_property_readonly("power",
                             [](py::object self) {
                               ColorFlowMap& c = self.cast<ColorFlowMap&>();
                               return view<real>(c.power.data(), {c.grid.n_axis0, c.grid.n_axis1},
                                                 self);
                             })
      .def_property_readonly("variance",
                             [](py::object self) {
                               ColorFlowMap& c = self.cast<ColorFlowMap&>();
                               return view<real>(c.variance.data(),
                                                 {c.grid.n_axis0, c.grid.n_axis1}, self);
                             })
      .def_property_readonly("valid", [](py::object self) {
        ColorFlowMap& c = self.cast<ColorFlowMap&>();
        return view<std::uint8_t>(c.valid.data(), {c.grid.n_axis0, c.grid.n_axis1}, self);
      });

  m.def("color_flow", &color_flow, py::arg("ensemble"), py::arg("prf"),
        py::arg("speed_of_sound"), py::arg("config") = DopplerConfig{},
        py::call_guard<py::gil_scoped_release>());

  m.def("apply_wall_filter", &apply_wall_filter, py::arg("ensemble"), py::arg("type"),
        py::arg("polynomial_order") = 1, py::call_guard<py::gil_scoped_release>());

  m.def("power_doppler",
        [](const std::vector<Frame>& ens, WallFilter type, int order) {
          std::vector<real> p = power_doppler(ens, type, order);
          return copy_vector(p, {ens.front().grid.n_axis0, ens.front().grid.n_axis1});
        },
        py::arg("ensemble"), py::arg("type") = WallFilter::PolynomialRegression,
        py::arg("polynomial_order") = 1);

  py::class_<DisplacementMap>(m, "DisplacementMap")
      .def_readonly("grid", &DisplacementMap::grid)
      .def_property_readonly("axial",
                             [](py::object self) {
                               DisplacementMap& d = self.cast<DisplacementMap&>();
                               return view<real>(d.axial.data(), {d.grid.n_axis0, d.grid.n_axis1},
                                                 self);
                             })
      .def_property_readonly("correlation", [](py::object self) {
        DisplacementMap& d = self.cast<DisplacementMap&>();
        return view<real>(d.correlation.data(), {d.grid.n_axis0, d.grid.n_axis1}, self);
      });

  m.def("estimate_axial_displacement", &estimate_axial_displacement, py::arg("before"),
        py::arg("after"), py::arg("speed_of_sound"), py::arg("kernel_axial") = 8,
        py::arg("kernel_lateral") = 2, py::arg("max_lag") = 8,
        py::call_guard<py::gil_scoped_release>());

  m.def("axial_strain",
        [](const DisplacementMap& d, int fit_length) {
          std::vector<real> s = axial_strain(d, fit_length);
          return copy_vector(s, {d.grid.n_axis0, d.grid.n_axis1});
        },
        py::arg("displacement"), py::arg("fit_length") = 15);

  // --- pipeline ------------------------------------------------------------
  py::class_<PipelineConfig>(m, "PipelineConfig")
      .def(py::init<>())
      .def_readwrite("demod", &PipelineConfig::demod)
      .def_readwrite("beamformer", &PipelineConfig::beamformer)
      .def_readwrite("dynamic_range_db", &PipelineConfig::dynamic_range_db)
      .def_readwrite("gain_db", &PipelineConfig::gain_db)
      .def_readwrite("apply_tgc", &PipelineConfig::apply_tgc)
      .def_readwrite("reference_smoothing", &PipelineConfig::reference_smoothing);

  py::class_<PipelineStats>(m, "PipelineStats")
      .def_readonly("demodulate_ms", &PipelineStats::demodulate_ms)
      .def_readonly("beamform_ms", &PipelineStats::beamform_ms)
      .def_readonly("postprocess_ms", &PipelineStats::postprocess_ms)
      .def_readonly("total_ms", &PipelineStats::total_ms)
      .def_property_readonly("fps", &PipelineStats::frames_per_second)
      .def("__repr__", [](const PipelineStats& s) {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "PipelineStats(demod=%.1fms, beamform=%.1fms, post=%.1fms, total=%.1fms, "
                      "%.1f fps)",
                      s.demodulate_ms, s.beamform_ms, s.postprocess_ms, s.total_ms,
                      s.frames_per_second());
        return std::string(buf);
      });

  py::class_<ImagingPipeline>(m, "ImagingPipeline")
      .def(py::init<ScanGrid, PipelineConfig>(), py::arg("grid"),
           py::arg("config") = PipelineConfig{})
      .def("process_frame", &ImagingPipeline::process_frame, py::arg("channel_data"),
           py::call_guard<py::gil_scoped_release>())
      .def("process_display",
           [](ImagingPipeline& p, const ChannelData& d) {
             std::vector<real> img;
             {
               py::gil_scoped_release release;
               img = p.process_display(d);
             }
             return copy_vector(img, {p.grid().n_axis0, p.grid().n_axis1});
           },
           py::arg("channel_data"))
      .def("to_display",
           [](ImagingPipeline& p, const Frame& f, const Medium& med) {
             std::vector<real> img;
             {
               py::gil_scoped_release release;
               img = p.to_display(f, med);
             }
             return copy_vector(img, {p.grid().n_axis0, p.grid().n_axis1});
           },
           py::arg("frame"), py::arg("medium") = Medium{})
      .def("reset", &ImagingPipeline::reset)
      .def_property_readonly("grid", &ImagingPipeline::grid)
      .def_property_readonly("stats", &ImagingPipeline::stats)
      .def_property_readonly("frame_count", &ImagingPipeline::frame_count)
      .def_property("config", py::overload_cast<>(&ImagingPipeline::config, py::const_),
                    [](ImagingPipeline& p, const PipelineConfig& c) { p.config() = c; });

  m.attr("__version__") = "0.1.0";
}
