#pragma once
//
// glb_core.h - the one definition of the glTF binaries VOLT writes and reads.
//
// Every model VOLT stores (trajectory frames, colour codings, particle filters, plugin exports,
// Python SDK output) is written by encode_points or encode_mesh, and every reader that needs the
// geometry back (the preview rasterizer, format migrations) goes through decode_glb. Shared by the
// Node (N-API) and Python (pybind11) bindings; no runtime headers here.
//
// The files are standard glTF 2.0 with two ratified extensions, so any glTF loader opens them:
//   - KHR_mesh_quantization: positions are unsigned 16-bit integers on a uniform grid of
//     POSITION_STEP angstrom (coarser only when the extent needs more than 65535 steps); the
//     node's translation and scale turn them back into coordinates. Normals are octahedral snorm8.
//     Colours are RGBA8.
//   - EXT_meshopt_compression: every buffer view is meshopt-encoded (vertex codec version 0).
// Points keep the order they are given in: point i of the file is atom i of the frame, so masks
// and per-atom data computed on the frame line up with the model without any permutation.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace glbcore {

// Grid spacing, in angstrom, of every position VOLT writes. Rounding moves a point by at most
// half of it, well below thermal motion and below what a rendered atom can show.
constexpr double POSITION_STEP = 0.01;

struct Material {
    float baseColor[4] = {1, 1, 1, 1};
    float metallic = 0;
    float roughness = 1;
    float emissive[3] = {0, 0, 0};
    bool doubleSided = true;
};

// Point cloud. positions: count * 3 floats. rgba: count * 4 bytes, or nullptr for white points.
std::vector<uint8_t> encode_points(const float* positions, size_t count, const uint8_t* rgba);

// Indexed triangle mesh. positions and normals: vertexCount * 3 floats. rgba: vertexCount * 4
// bytes or nullptr. indexCount must be a multiple of 3. Triangles keep their order, so ranges of
// triangles stay valid; vertices are renumbered by first use and unreferenced ones are dropped.
std::vector<uint8_t> encode_mesh(
    const float* positions,
    const float* normals,
    size_t vertexCount,
    const uint8_t* rgba,
    const uint32_t* indices,
    size_t indexCount,
    const Material& material
);

enum class PrimitiveMode { Points, Triangles };

struct Primitive {
    PrimitiveMode mode = PrimitiveMode::Points;
    std::vector<float> positions;   // world space, 3 per vertex
    std::vector<float> normals;     // world space, 3 per vertex; empty when absent
    std::vector<uint8_t> colors;    // RGBA8, 4 per vertex; empty when absent
    std::vector<uint32_t> indices;  // empty for points
    Material material;
};

// Reads the point and triangle primitives of any glTF binary (the files above, and the plain
// float32 ones VOLT wrote before them). Returns false and sets error when the data is not usable.
bool decode_glb(const uint8_t* data, size_t size, std::vector<Primitive>& out, std::string& error);

// Converts colours in [0, 1] with 3 (RGB) or 4 (RGBA) components per item to RGBA8.
void rgba8_from_floats(const float* colors, size_t count, size_t components, uint8_t* rgba);

}  // namespace glbcore
