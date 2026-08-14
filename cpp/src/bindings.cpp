#include "rmsynth_reference/core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace rmsynth_reference;

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native core for RMSynth Reference Edition";

    py::register_exception<validation_error>(module, "ValidationError", PyExc_ValueError);
    py::register_exception<resource_limit_error>(module, "ResourceLimitError", PyExc_RuntimeError);

    py::class_<decode_result>(module, "DecodeResult")
        .def_readonly("codeword", &decode_result::codeword)
        .def_readonly("selected_terms", &decode_result::selected_terms)
        .def_readonly("distance", &decode_result::distance)
        .def_readonly("candidates", &decode_result::candidates)
        .def_readonly("ties", &decode_result::ties);

    module.def("rm_dimension", &rm_dimension, py::arg("qubits"), py::arg("order"));
    module.def("rm_basis_terms", &rm_basis_terms, py::arg("qubits"), py::arg("order"));
    module.def("rm_generator_rows", &rm_generator_rows, py::arg("qubits"), py::arg("order"));
    module.def("decode_exact", &decode_exact, py::arg("received"), py::arg("qubits"),
               py::arg("order"));
    module.def("synthesize_linear_map", &synthesize_linear_map, py::arg("rows"));
}
