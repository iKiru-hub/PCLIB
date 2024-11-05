#include "pcnn.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {

    // PCNN
    py::class_<pcNN>(m, "pcNN")
        .def(py::init<int, double, double>(),
             py::arg("N"),
             py::arg("gain"),
             py::arg("offset"))
        .def("call", &pcNN::call)
        .def_property_readonly("N", &pcNN::get_N)
        .def_property_readonly("gain", &pcNN::get_gain)
        .def_property_readonly("offset", &pcNN::get_offset)
        .def_property("W",
            &pcNN::get_W,
            &pcNN::set_W)
        .def_property("C",
            &pcNN::get_C,
            &pcNN::set_C)
        .def_property("mask",
            &pcNN::get_mask,
            &pcNN::set_mask)
        .def_property("u",
            &pcNN::get_u,
            &pcNN::set_u);

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
        .def("info", &LeakyVariable1D::info)
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
        .def("info", &LeakyVariableND::info)
        .def("get_v", &LeakyVariableND::get_v);
}
