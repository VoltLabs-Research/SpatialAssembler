// Python (pybind11) binding over the shared glb_core.h. Mirrors the Node addon
// using the same C++ encoders, so there is one source of truth for GLB output.
#include <pybind11/pybind11.h>

#include <cstdint>
#include <vector>

#include "glb_core.h"

namespace py = pybind11;

static const uint8_t* buf_bytes(const py::buffer& b, size_t& nbytes) {
    py::buffer_info info = b.request();
    nbytes = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
    return reinterpret_cast<const uint8_t*>(info.ptr);
}

static void read_doubles(const py::sequence& s, double* out, size_t n) {
    if (static_cast<size_t>(py::len(s)) < n) throw py::value_error("sequence too short");
    for (size_t i = 0; i < n; i++) out[i] = py::cast<double>(s[i]);
}

static py::bytes to_bytes(const std::vector<uint8_t>& v) {
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}

static py::bytes atom_glb(py::buffer positions, py::buffer types, py::sequence vmin, py::sequence vmax) {
    size_t pb, tb;
    const float* pos = reinterpret_cast<const float*>(buf_bytes(positions, pb));
    const uint16_t* typ = reinterpret_cast<const uint16_t*>(buf_bytes(types, tb));
    double mn[3], mx[3];
    read_doubles(vmin, mn, 3);
    read_doubles(vmax, mx, 3);
    return to_bytes(glbcore::generate_atom_glb(pos, (pb / 4) / 3, typ, mn, mx));
}

static py::bytes point_cloud_glb(py::buffer positions, py::buffer colors, py::sequence vmin, py::sequence vmax) {
    size_t pb, cb;
    const float* pos = reinterpret_cast<const float*>(buf_bytes(positions, pb));
    const float* col = reinterpret_cast<const float*>(buf_bytes(colors, cb));
    double mn[3], mx[3];
    read_doubles(vmin, mn, 3);
    read_doubles(vmax, mx, 3);
    return to_bytes(glbcore::generate_point_cloud_glb(pos, pb / 4, col, cb / 4, mn, mx));
}

static py::bytes mesh_glb(py::buffer positions, py::buffer normals, py::buffer indices,
                          py::sequence bounds, py::sequence base_color, double metallic, double roughness,
                          py::sequence emissive, bool double_sided, py::object colors) {
    size_t pb, nb, ib;
    const float* pos = reinterpret_cast<const float*>(buf_bytes(positions, pb));
    const float* nor = reinterpret_cast<const float*>(buf_bytes(normals, nb));
    const uint32_t* idx = reinterpret_cast<const uint32_t*>(buf_bytes(indices, ib));

    const float* col = nullptr;
    size_t cb = 0;
    if (!colors.is_none()) {
        col = reinterpret_cast<const float*>(buf_bytes(py::cast<py::buffer>(colors), cb));
    }

    double bnd[6];
    read_doubles(bounds, bnd, 6);
    glbcore::Material mat;
    read_doubles(base_color, mat.baseColor, 4);
    read_doubles(emissive, mat.emissive, 3);
    mat.metallic = metallic;
    mat.roughness = roughness;
    mat.doubleSided = double_sided;

    return to_bytes(glbcore::generate_mesh_glb(pos, pb / 4, nor, nb / 4, idx, ib / 4, col, cb / 4, bnd, mat));
}

PYBIND11_MODULE(_spatial_assembler, m) {
    m.doc() = "Native GLB assembler (shared C++ core with the Node addon).";
    m.def("atom_glb", &atom_glb, py::arg("positions"), py::arg("types"), py::arg("vmin"), py::arg("vmax"));
    m.def("point_cloud_glb", &point_cloud_glb, py::arg("positions"), py::arg("colors"), py::arg("vmin"), py::arg("vmax"));
    m.def("mesh_glb", &mesh_glb, py::arg("positions"), py::arg("normals"), py::arg("indices"), py::arg("bounds"),
          py::arg("base_color"), py::arg("metallic"), py::arg("roughness"), py::arg("emissive"),
          py::arg("double_sided"), py::arg("colors") = py::none());
}
