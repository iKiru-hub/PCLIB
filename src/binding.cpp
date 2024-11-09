#include "pcnn.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

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
        .def(py::init<std::string, float, float>(),
             py::arg("name"),
             py::arg("eq"),
             py::arg("tau"))
        .def("__call__", &LeakyVariable1D::call,
             py::arg("x") = 0.0)
        .def("__str__", &LeakyVariable1D::str)
        .def("__len__", &LeakyVariable1D::len)
        .def("__repr__", &LeakyVariable1D::repr)
        .def("get_v", &LeakyVariable1D::get_v);

    // LeakyVariable ND
    py::class_<LeakyVariableND>(m, "LeakyVariableND")
        .def(py::init<std::string, float, float, size_t>(),
             py::arg("name"),
             py::arg("eq"),
             py::arg("tau"),
             py::arg("ndim"))
        .def("__call__", &LeakyVariableND::call,
             py::arg("x"))
        .def("__str__", &LeakyVariableND::str)
        .def("__len__", &LeakyVariableND::len)
        .def("print_v", &LeakyVariableND::print_v)
        .def("__repr__", &LeakyVariableND::repr)
        .def("get_v", &LeakyVariableND::get_v);

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
        .def(py::init<int, int, float, float, float, float, \
             int, float, PCLayer, std::string>(),
             py::arg("N"),
             py::arg("Nj"),
             py::arg("gain"),
             py::arg("offset"),
             py::arg("rep_threshold"),
             py::arg("rec_threshold"),
             py::arg("num_neighbors"),
             py::arg("trace_tau"),
             py::arg("xfilter"),
             py::arg("name"))
        .def("__call__", &PCNN::call,
             py::arg("x"))
        .def("__str__", &PCNN::str)
        .def("__len__", &PCNN::len)
        .def("__repr__", &PCNN::repr);
}
