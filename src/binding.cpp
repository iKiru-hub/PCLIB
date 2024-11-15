#include "pcnn.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

    // function `set_debug`
    m.def("set_debug", &set_debug,
          py::arg("flag"));

    // Sampling Module
    py::class_<SamplingModule>(m, "SamplingModule")
        .def(py::init<std::string, float>(),
             py::arg("name"),
             py::arg("speed"))
        .def("__call__", &SamplingModule::call,
             py::arg("keep") = false)
        .def("update", &SamplingModule::update)
        .def("__len__", &SamplingModule::len)
        .def("__str__", &SamplingModule::str)
        .def("reset", &SamplingModule::reset)
        .def("is_done", &SamplingModule::is_done)
        .def("get_idx", &SamplingModule::get_idx)
        .def("get_counter", &SamplingModule::get_counter);

    // LeakyVariable 1D
    py::class_<LeakyVariable1D>(m, "LeakyVariable1D")
        .def(py::init<std::string, float, float,
             float>(),
             py::arg("name"),
             py::arg("eq"),
             py::arg("tau"),
             py::arg("min_v") = 0.0)
        .def("__call__", &LeakyVariable1D::call,
             py::arg("x") = 0.0,
             py::arg("simulate") = false)
        .def("__str__", &LeakyVariable1D::str)
        .def("__len__", &LeakyVariable1D::len)
        .def("__repr__", &LeakyVariable1D::repr)
        .def("get_v", &LeakyVariable1D::get_v)
        .def("get_name", &LeakyVariable1D::get_name)
        .def("set_eq", &LeakyVariable1D::set_eq);

    // LeakyVariable ND
    py::class_<LeakyVariableND>(m, "LeakyVariableND")
        .def(py::init<std::string, float, float, int,
             float>(),
             py::arg("name"),
             py::arg("eq"),
             py::arg("tau"),
             py::arg("ndim"),
             py::arg("min_v") = 0.0)
        .def("__call__", &LeakyVariableND::call,
             py::arg("x"),
             py::arg("simulate") = false)
        .def("__str__", &LeakyVariableND::str)
        .def("__repr__", &LeakyVariableND::repr)
        .def("__len__", &LeakyVariableND::len)
        .def("print_v", &LeakyVariableND::print_v)
        .def("get_v", &LeakyVariableND::get_v)
        .def("set_eq", &LeakyVariableND::set_eq);

    // (InputFilter) Place Cell Layer
    py::class_<PCLayer>(m, "PCLayer")
        .def(py::init<int, float, std::array<float, 4>>(),
             py::arg("n"),
             py::arg("sigma"),
             py::arg("bounds"))
        .def("__call__", &PCLayer::call,
             py::arg("x"))
        .def("__str__", &PCLayer::str)
        .def("__len__", &PCLayer::len)
        .def("__repr__", &PCLayer::repr);

    // PCNN network model
    py::class_<PCNN>(m, "PCNN")
        .def(py::init<int, int, float, float,
             float, float, float, float, \
             int, float, PCLayer, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain"),
             py::arg("offset"),
             py::arg("clip_min"),
             py::arg("threshold"),
             py::arg("rep_threshold"),
             py::arg("rec_threshold"),
             py::arg("num_neighbors"),
             py::arg("trace_tau"),
             py::arg("xfilter"),
             py::arg("name"))
        .def("__call__", &PCNN::call,
             py::arg("x"),
             py::arg("frozen") = false,
             py::arg("traced") = true)
        .def("__str__", &PCNN::str)
        .def("__len__", &PCNN::len)
        .def("__repr__", &PCNN::repr)
        .def("get_size", &PCNN::get_size)
        .def("get_trace", &PCNN::get_trace)
        .def("get_wff", &PCNN::get_wff)
        .def("get_wrec", &PCNN::get_wrec)
        .def("get_connectivity", &PCNN::get_connectivity)
        .def("get_delta_update", &PCNN::get_delta_update)
        .def("get_centers", &PCNN::get_centers)
        .def("fwd_ext", &PCNN::fwd_ext,
             py::arg("x"))
        .def("fwd_int", &PCNN::fwd_int,
             py::arg("a"));
}
