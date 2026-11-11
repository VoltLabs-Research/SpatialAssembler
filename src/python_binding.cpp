// python_binding.cpp - pybind11 surface of the shared core, mirroring the Node addon.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "colors.h"
#include "glb_core.h"

namespace py = pybind11;

namespace {

struct View {
    const uint8_t* data = nullptr;
    size_t items = 0;
    std::string format;
};

View view_of(const py::buffer& buffer) {
    py::buffer_info info = buffer.request();
    View view;
    view.data = static_cast<const uint8_t*>(info.ptr);
    view.items = static_cast<size_t>(info.size);
    view.format = info.format;
    if (static_cast<size_t>(info.itemsize) == 1) view.format = "B";
    return view;
}

bool is_float32(const View& view) { return view.format == "f" || view.format == "<f" || view.format == "=f"; }

std::vector<float> floats_of(const py::buffer& buffer, const char* name) {
    View view = view_of(buffer);
    if (!is_float32(view)) throw py::type_error(std::string(name) + " must be a float32 buffer");
    const float* data = reinterpret_cast<const float*>(view.data);
    return std::vector<float>(data, data + view.items);
}

// RGBA8 bytes (4 per item) or float32 in [0, 1] (3 or 4 per item); None for no colours.
bool colors_of(const py::object& colors, size_t count, std::vector<uint8_t>& rgba) {
    if (colors.is_none()) return false;
    View view = view_of(py::cast<py::buffer>(colors));
    if (!is_float32(view) && view.items == count * 4) {
        rgba.assign(view.data, view.data + count * 4);
        return true;
    }
    if (is_float32(view) && (view.items == count * 3 || view.items == count * 4)) {
        rgba.resize(count * 4);
        glbcore::rgba8_from_floats(reinterpret_cast<const float*>(view.data), count, view.items / (count ? count : 1), rgba.data());
        return true;
    }
    throw py::value_error("colors must be RGBA8 bytes or RGB/RGBA float32 values, one per vertex");
}

glbcore::Material material_of(const py::object& source) {
    glbcore::Material material;
    if (source.is_none()) return material;
    py::dict dict = py::cast<py::dict>(source);
    auto read = [&](const char* key, float* out, size_t count) {
        if (!dict.contains(key)) return;
        py::sequence values = py::cast<py::sequence>(dict[key]);
        for (size_t i = 0; i < count && i < static_cast<size_t>(py::len(values)); i++) out[i] = py::cast<float>(values[i]);
    };
    read("baseColor", material.baseColor, 4);
    read("emissive", material.emissive, 3);
    if (dict.contains("metallic")) material.metallic = py::cast<float>(dict["metallic"]);
    if (dict.contains("roughness")) material.roughness = py::cast<float>(dict["roughness"]);
    if (dict.contains("doubleSided")) material.doubleSided = py::cast<bool>(dict["doubleSided"]);
    return material;
}

py::bytes bytes_of(const std::vector<uint8_t>& value) {
    return py::bytes(reinterpret_cast<const char*>(value.data()), value.size());
}

py::bytes encode_point_cloud(py::buffer positions, py::object colors) {
    std::vector<float> xyz = floats_of(positions, "positions");
    if (xyz.size() % 3 != 0) throw py::value_error("positions must hold xyz triples");
    size_t count = xyz.size() / 3;
    std::vector<uint8_t> rgba;
    bool hasColors = colors_of(colors, count, rgba);
    return bytes_of(glbcore::encode_points(xyz.data(), count, hasColors ? rgba.data() : nullptr));
}

py::bytes encode_mesh(py::buffer positions, py::buffer normals, py::buffer indices, py::object colors, py::object material) {
    std::vector<float> xyz = floats_of(positions, "positions");
    std::vector<float> nxyz = floats_of(normals, "normals");
    if (xyz.size() % 3 != 0 || nxyz.size() != xyz.size()) throw py::value_error("positions and normals must hold matching xyz triples");
    size_t vertexCount = xyz.size() / 3;

    View index = view_of(indices);
    std::vector<uint32_t> wide(index.items);
    py::buffer_info info = indices.request();
    for (size_t i = 0; i < index.items; i++) {
        if (info.itemsize == 4) wide[i] = reinterpret_cast<const uint32_t*>(index.data)[i];
        else if (info.itemsize == 2) wide[i] = reinterpret_cast<const uint16_t*>(index.data)[i];
        else throw py::type_error("indices must be a uint16 or uint32 buffer");
    }

    std::vector<uint8_t> rgba;
    bool hasColors = colors_of(colors, vertexCount, rgba);
    return bytes_of(glbcore::encode_mesh(xyz.data(), nxyz.data(), vertexCount, hasColors ? rgba.data() : nullptr,
        wide.data(), wide.size(), material_of(material)));
}

py::list decode_glb(py::bytes data) {
    std::string raw = data;
    std::vector<glbcore::Primitive> primitives;
    std::string error;
    if (!glbcore::decode_glb(reinterpret_cast<const uint8_t*>(raw.data()), raw.size(), primitives, error)) {
        throw py::value_error(error);
    }
    auto as_bytes = [](const void* data, size_t size) {
        return py::bytes(reinterpret_cast<const char*>(data), size);
    };
    py::list out;
    for (const glbcore::Primitive& primitive : primitives) {
        py::dict item;
        item["mode"] = primitive.mode == glbcore::PrimitiveMode::Points ? "points" : "triangles";
        item["positions"] = as_bytes(primitive.positions.data(), primitive.positions.size() * sizeof(float));
        item["normals"] = primitive.normals.empty() ? py::object(py::none()) : as_bytes(primitive.normals.data(), primitive.normals.size() * sizeof(float));
        item["colors"] = primitive.colors.empty() ? py::object(py::none()) : as_bytes(primitive.colors.data(), primitive.colors.size());
        item["indices"] = primitive.indices.empty() ? py::object(py::none()) : as_bytes(primitive.indices.data(), primitive.indices.size() * sizeof(uint32_t));
        py::dict material;
        const glbcore::Material& m = primitive.material;
        material["baseColor"] = py::make_tuple(m.baseColor[0], m.baseColor[1], m.baseColor[2], m.baseColor[3]);
        material["emissive"] = py::make_tuple(m.emissive[0], m.emissive[1], m.emissive[2]);
        material["metallic"] = m.metallic;
        material["roughness"] = m.roughness;
        material["doubleSided"] = m.doubleSided;
        item["material"] = material;
        out.append(item);
    }
    return out;
}

py::bytes colors_by_type(py::buffer types) {
    py::buffer_info info = types.request();
    if (info.itemsize != 2) throw py::type_error("types must be a uint16 buffer");
    size_t count = static_cast<size_t>(info.size);
    std::vector<uint8_t> rgba(count * 4);
    glbcore::colors_by_type(static_cast<const uint16_t*>(info.ptr), count, rgba.data());
    return bytes_of(rgba);
}

}  // namespace

PYBIND11_MODULE(_spatial_assembler, m) {
    m.doc() = "VOLT glTF encoder/decoder (shared C++ core with the Node addon).";
    m.attr("POSITION_STEP") = glbcore::POSITION_STEP;
    m.def("encode_point_cloud", &encode_point_cloud, py::arg("positions"), py::arg("colors") = py::none(),
          "Encode a point cloud. positions: float32 xyz; colors: RGBA8 bytes or RGB/RGBA float32, or None.");
    m.def("encode_mesh", &encode_mesh, py::arg("positions"), py::arg("normals"), py::arg("indices"),
          py::arg("colors") = py::none(), py::arg("material") = py::none(),
          "Encode an indexed triangle mesh.");
    m.def("decode_glb", &decode_glb, py::arg("data"),
          "Decode the point and triangle primitives of a glTF binary into raw little-endian buffers.");
    m.def("colors_by_type", &colors_by_type, py::arg("types"), "RGBA8 colours of atom types (uint16).");
}
