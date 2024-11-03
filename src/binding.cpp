#include "pcnn.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pclib, m) {
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
}
