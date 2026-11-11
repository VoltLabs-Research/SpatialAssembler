// glb_core.cpp - encoder and decoder for the glTF binaries described in glb_core.h.
#include "glb_core.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "meshoptimizer.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

namespace glbcore {
namespace {

constexpr uint32_t GLB_MAGIC = 0x46546C67;
constexpr uint32_t GLB_CHUNK_JSON = 0x4E4F534A;
constexpr uint32_t GLB_CHUNK_BIN = 0x004E4942;
constexpr uint32_t TARGET_VERTICES = 34962;
constexpr uint32_t TARGET_INDICES = 34963;
constexpr int VERTEX_CODEC_VERSION = 0;
constexpr int NORMAL_BITS = 8;
constexpr size_t QUANTIZED_MAX = 65535;

const char* const EXTENSIONS =
    R"("extensionsUsed":["EXT_meshopt_compression","KHR_mesh_quantization"],)"
    R"("extensionsRequired":["EXT_meshopt_compression","KHR_mesh_quantization"],)";

void append(std::string& out, const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    int written = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (written < 0) return;
    if ((size_t)written < sizeof(buffer)) {
        out.append(buffer, (size_t)written);
        return;
    }
    std::vector<char> large((size_t)written + 1);
    va_start(args, format);
    vsnprintf(large.data(), large.size(), format, args);
    va_end(args);
    out.append(large.data(), (size_t)written);
}

void pad_to_four(std::vector<uint8_t>& bytes, uint8_t value) {
    while (bytes.size() % 4 != 0) bytes.push_back(value);
}

std::vector<uint8_t> assemble(const std::string& json, const std::vector<uint8_t>& bin) {
    std::vector<uint8_t> jsonChunk(json.begin(), json.end());
    pad_to_four(jsonChunk, ' ');
    std::vector<uint8_t> binChunk(bin);
    pad_to_four(binChunk, 0);

    size_t total = 12 + 8 + jsonChunk.size() + (binChunk.empty() ? 0 : 8 + binChunk.size());
    std::vector<uint8_t> glb;
    glb.reserve(total);
    auto put = [&](uint32_t value) {
        uint8_t raw[4];
        std::memcpy(raw, &value, 4);
        glb.insert(glb.end(), raw, raw + 4);
    };
    put(GLB_MAGIC);
    put(2);
    put((uint32_t)total);
    put((uint32_t)jsonChunk.size());
    put(GLB_CHUNK_JSON);
    glb.insert(glb.end(), jsonChunk.begin(), jsonChunk.end());
    if (!binChunk.empty()) {
        put((uint32_t)binChunk.size());
        put(GLB_CHUNK_BIN);
        glb.insert(glb.end(), binChunk.begin(), binChunk.end());
    }
    return glb;
}

std::vector<uint8_t> empty_glb() {
    return assemble(R"({"asset":{"version":"2.0","generator":"VOLT"},"scene":0,"scenes":[{}]})", {});
}

// Uniform grid the positions are rounded to. One step for every axis keeps the node scale
// uniform, so normals need no correction and the dequantized model is the original geometry.
struct Grid {
    double origin[3] = {0, 0, 0};
    double step = POSITION_STEP;
};

Grid grid_for(const float* positions, size_t count) {
    double lo[3] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    double hi[3] = {-lo[0], -lo[1], -lo[2]};
    for (size_t i = 0; i < count; i++) {
        for (int a = 0; a < 3; a++) {
            float v = positions[i * 3 + a];
            if (!std::isfinite(v)) continue;
            lo[a] = std::min(lo[a], (double)v);
            hi[a] = std::max(hi[a], (double)v);
        }
    }

    Grid grid;
    double extent = 0;
    for (int a = 0; a < 3; a++) {
        if (lo[a] > hi[a]) lo[a] = hi[a] = 0;
        grid.origin[a] = lo[a];
        extent = std::max(extent, hi[a] - lo[a]);
    }
    grid.step = std::max(POSITION_STEP, extent / (double)QUANTIZED_MAX);
    return grid;
}

// Positions as u16x4 (w = 0: meshopt needs a 4-byte multiple stride, the accessor reads xyz).
std::vector<uint16_t> quantize_positions(const float* positions, size_t count, const Grid& grid, uint16_t max[3]) {
    std::vector<uint16_t> out(count * 4, 0);
    max[0] = max[1] = max[2] = 0;
    const double inv = 1.0 / grid.step;
    for (size_t i = 0; i < count; i++) {
        for (int a = 0; a < 3; a++) {
            float v = positions[i * 3 + a];
            double q = std::isfinite(v) ? std::round(((double)v - grid.origin[a]) * inv) : 0.0;
            uint16_t value = (uint16_t)std::min((double)QUANTIZED_MAX, std::max(0.0, q));
            out[i * 4 + a] = value;
            max[a] = std::max(max[a], value);
        }
    }
    return out;
}

std::vector<uint8_t> encode_attributes(const void* data, size_t count, size_t stride) {
    std::vector<uint8_t> out(meshopt_encodeVertexBufferBound(count, stride));
    out.resize(meshopt_encodeVertexBufferLevel(out.data(), out.size(), data, count, stride, 2, VERTEX_CODEC_VERSION));
    return out;
}

std::vector<uint8_t> encode_triangles(const std::vector<uint32_t>& indices, size_t vertexCount) {
    std::vector<uint8_t> out(meshopt_encodeIndexBufferBound(indices.size(), vertexCount));
    out.resize(meshopt_encodeIndexBuffer(out.data(), out.size(), indices.data(), indices.size()));
    return out;
}

// Buffer 0 is the GLB BIN chunk holding the compressed streams; buffer 1 is the decoded
// layout the extension fills in, which has no bytes of its own.
struct Layout {
    std::vector<uint8_t> bin;
    size_t decodedSize = 0;
    std::string views;

    void add(std::vector<uint8_t> compressed, size_t count, size_t stride, uint32_t target, const char* mode, const char* filter) {
        pad_to_four(bin, 0);
        size_t compressedOffset = bin.size();
        bin.insert(bin.end(), compressed.begin(), compressed.end());

        decodedSize = (decodedSize + 3) & ~(size_t)3;
        size_t decodedOffset = decodedSize;
        decodedSize += count * stride;

        if (!views.empty()) views += ",";
        append(views, R"({"buffer":1,"byteOffset":%zu,"byteLength":%zu,)", decodedOffset, count * stride);
        if (target == TARGET_VERTICES) append(views, R"("byteStride":%zu,)", stride);
        append(views,
            R"("target":%u,"extensions":{"EXT_meshopt_compression":{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"byteStride":%zu,"count":%zu,"mode":"%s")",
            target, compressedOffset, compressed.size(), stride, count, mode);
        if (filter) append(views, R"(,"filter":"%s")", filter);
        views += "}}}";
    }

    std::string buffers() const {
        std::string out;
        append(out,
            R"("buffers":[{"byteLength":%zu},{"byteLength":%zu,"extensions":{"EXT_meshopt_compression":{"fallback":true}}}])",
            bin.size(), decodedSize);
        return out;
    }
};

std::string grid_node(const char* name, const Grid& grid) {
    std::string out;
    append(out,
        R"("nodes":[{"mesh":0,"name":"%s","translation":[%.17g,%.17g,%.17g],"scale":[%.17g,%.17g,%.17g]}],)",
        name, grid.origin[0], grid.origin[1], grid.origin[2], grid.step, grid.step, grid.step);
    return out;
}

std::string position_accessor(size_t count, const uint16_t max[3]) {
    std::string out;
    append(out,
        R"({"bufferView":0,"componentType":5123,"count":%zu,"type":"VEC3","min":[0,0,0],"max":[%u,%u,%u]})",
        count, (unsigned)max[0], (unsigned)max[1], (unsigned)max[2]);
    return out;
}

}  // namespace

void rgba8_from_floats(const float* colors, size_t count, size_t components, uint8_t* rgba) {
    for (size_t i = 0; i < count; i++) {
        for (size_t c = 0; c < 4; c++) {
            float v = c < components ? colors[i * components + c] : 1.0f;
            if (!std::isfinite(v)) v = c == 3 ? 1.0f : 0.0f;
            rgba[i * 4 + c] = (uint8_t)std::lround(std::max(0.0f, std::min(1.0f, v)) * 255.0f);
        }
    }
}

std::vector<uint8_t> encode_points(const float* positions, size_t count, const uint8_t* rgba) {
    if (count == 0) return empty_glb();

    const Grid grid = grid_for(positions, count);
    uint16_t max[3];
    const std::vector<uint16_t> quantized = quantize_positions(positions, count, grid, max);
    std::vector<uint8_t> colors(count * 4, 255);
    if (rgba) std::memcpy(colors.data(), rgba, count * 4);

    Layout layout;
    layout.add(encode_attributes(quantized.data(), count, 8), count, 8, TARGET_VERTICES, "ATTRIBUTES", nullptr);
    layout.add(encode_attributes(colors.data(), count, 4), count, 4, TARGET_VERTICES, "ATTRIBUTES", nullptr);

    std::string json = R"({"asset":{"version":"2.0","generator":"VOLT"},)";
    json += EXTENSIONS;
    json += R"("scene":0,"scenes":[{"nodes":[0]}],)";
    json += grid_node("Atoms", grid);
    json += R"("meshes":[{"name":"AtomCloud","primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"mode":0}]}],)";
    json += R"("accessors":[)" + position_accessor(count, max);
    append(json, R"(,{"bufferView":1,"componentType":5121,"normalized":true,"count":%zu,"type":"VEC4"}],)", count);
    json += R"("bufferViews":[)" + layout.views + "],";
    json += layout.buffers() + "}";
    return assemble(json, layout.bin);
}

std::vector<uint8_t> encode_mesh(
    const float* positions,
    const float* normals,
    size_t vertexCount,
    const uint8_t* rgba,
    const uint32_t* indices,
    size_t indexCount,
    const Material& material
) {
    indexCount -= indexCount % 3;
    if (vertexCount == 0 || indexCount == 0) return empty_glb();

    // Triangles keep their order (sidecars address them by range); vertices are renumbered in the
    // order the triangles first use them, which is what lets the vertex codec find small deltas.
    std::vector<uint32_t> order(indices, indices + indexCount);
    for (uint32_t& index : order) {
        if (index >= vertexCount) index = 0;
    }
    std::vector<uint32_t> remap(vertexCount);
    const size_t uniqueCount = meshopt_optimizeVertexFetchRemap(remap.data(), order.data(), indexCount, vertexCount);
    for (uint32_t& index : order) index = remap[index];

    std::vector<float> remappedPositions(uniqueCount * 3);
    std::vector<float> octahedral(uniqueCount * 4, 0.0f);
    std::vector<uint8_t> colors(rgba ? uniqueCount * 4 : 0);
    for (size_t source = 0; source < vertexCount; source++) {
        uint32_t target = remap[source];
        if (target == ~0u) continue;
        std::memcpy(&remappedPositions[target * 3], &positions[source * 3], 3 * sizeof(float));

        float nx = normals ? normals[source * 3] : 0.0f;
        float ny = normals ? normals[source * 3 + 1] : 0.0f;
        float nz = normals ? normals[source * 3 + 2] : 1.0f;
        float length = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!std::isfinite(length) || length == 0.0f) {
            nx = 0.0f;
            ny = 0.0f;
            nz = 1.0f;
            length = 1.0f;
        }
        octahedral[target * 4] = nx / length;
        octahedral[target * 4 + 1] = ny / length;
        octahedral[target * 4 + 2] = nz / length;

        if (rgba) std::memcpy(&colors[target * 4], &rgba[source * 4], 4);
    }

    const Grid grid = grid_for(remappedPositions.data(), uniqueCount);
    uint16_t max[3];
    const std::vector<uint16_t> quantized = quantize_positions(remappedPositions.data(), uniqueCount, grid, max);
    std::vector<int8_t> packedNormals(uniqueCount * 4);
    meshopt_encodeFilterOct(packedNormals.data(), uniqueCount, 4, NORMAL_BITS, octahedral.data());

    const bool wideIndices = uniqueCount > 65535;
    Layout layout;
    layout.add(encode_attributes(quantized.data(), uniqueCount, 8), uniqueCount, 8, TARGET_VERTICES, "ATTRIBUTES", nullptr);
    layout.add(encode_attributes(packedNormals.data(), uniqueCount, 4), uniqueCount, 4, TARGET_VERTICES, "ATTRIBUTES", "OCTAHEDRAL");
    if (rgba) {
        layout.add(encode_attributes(colors.data(), uniqueCount, 4), uniqueCount, 4, TARGET_VERTICES, "ATTRIBUTES", nullptr);
    }
    const size_t indexView = rgba ? 3 : 2;
    layout.add(encode_triangles(order, uniqueCount), indexCount, wideIndices ? 4 : 2, TARGET_INDICES, "TRIANGLES", nullptr);

    std::string json = R"({"asset":{"version":"2.0","generator":"VOLT"},)";
    json += EXTENSIONS;
    json += R"("scene":0,"scenes":[{"nodes":[0]}],)";
    json += grid_node("Mesh", grid);
    append(json,
        R"("materials":[{"pbrMetallicRoughness":{"baseColorFactor":[%.6g,%.6g,%.6g,%.6g],"metallicFactor":%.6g,"roughnessFactor":%.6g},"emissiveFactor":[%.6g,%.6g,%.6g],"doubleSided":%s}],)",
        material.baseColor[0], material.baseColor[1], material.baseColor[2], material.baseColor[3],
        material.metallic, material.roughness,
        material.emissive[0], material.emissive[1], material.emissive[2],
        material.doubleSided ? "true" : "false");
    if (rgba) {
        append(json, R"("meshes":[{"name":"MeshGeometry","primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"COLOR_0":2},"indices":%zu,"material":0,"mode":4}]}],)", indexView);
    } else {
        append(json, R"("meshes":[{"name":"MeshGeometry","primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":%zu,"material":0,"mode":4}]}],)", indexView);
    }
    json += R"("accessors":[)" + position_accessor(uniqueCount, max);
    append(json, R"(,{"bufferView":1,"componentType":5120,"normalized":true,"count":%zu,"type":"VEC3"})", uniqueCount);
    if (rgba) {
        append(json, R"(,{"bufferView":2,"componentType":5121,"normalized":true,"count":%zu,"type":"VEC4"})", uniqueCount);
    }
    append(json, R"(,{"bufferView":%zu,"componentType":%u,"count":%zu,"type":"SCALAR"}],)",
        indexView, wideIndices ? 5125u : 5123u, indexCount);
    json += R"("bufferViews":[)" + layout.views + "],";
    json += layout.buffers() + "}";
    return assemble(json, layout.bin);
}

namespace {

bool decode_meshopt_views(cgltf_data* gltf, std::string& error) {
    for (cgltf_size i = 0; i < gltf->buffer_views_count; i++) {
        cgltf_buffer_view& view = gltf->buffer_views[i];
        if (!view.has_meshopt_compression) continue;

        const cgltf_meshopt_compression& mc = view.meshopt_compression;
        if (!mc.buffer || !mc.buffer->data) {
            error = "meshopt buffer view has no source data";
            return false;
        }
        const unsigned char* source = (const unsigned char*)mc.buffer->data + mc.offset;
        // cgltf_free releases view.data with free().
        void* target = std::malloc(mc.count * mc.stride);
        if (!target) {
            error = "out of memory decoding a buffer view";
            return false;
        }
        view.data = target;

        int rc = -1;
        switch (mc.mode) {
            case cgltf_meshopt_compression_mode_attributes:
                rc = meshopt_decodeVertexBuffer(target, mc.count, mc.stride, source, mc.size);
                break;
            case cgltf_meshopt_compression_mode_triangles:
                rc = meshopt_decodeIndexBuffer(target, mc.count, mc.stride, source, mc.size);
                break;
            case cgltf_meshopt_compression_mode_indices:
                rc = meshopt_decodeIndexSequence(target, mc.count, mc.stride, source, mc.size);
                break;
            default:
                break;
        }
        if (rc != 0) {
            error = "meshopt stream could not be decoded";
            return false;
        }

        switch (mc.filter) {
            case cgltf_meshopt_compression_filter_octahedral:
                meshopt_decodeFilterOct(target, mc.count, mc.stride);
                break;
            case cgltf_meshopt_compression_filter_quaternion:
                meshopt_decodeFilterQuat(target, mc.count, mc.stride);
                break;
            case cgltf_meshopt_compression_filter_exponential:
                meshopt_decodeFilterExp(target, mc.count, mc.stride);
                break;
            default:
                break;
        }
    }
    return true;
}

std::vector<float> unpack(const cgltf_accessor* accessor, size_t components) {
    std::vector<float> out(accessor->count * components, 0.0f);
    if (cgltf_num_components(accessor->type) == components) {
        cgltf_accessor_unpack_floats(accessor, out.data(), out.size());
        return out;
    }
    std::vector<float> element(16);
    for (cgltf_size i = 0; i < accessor->count; i++) {
        cgltf_accessor_read_float(accessor, i, element.data(), element.size());
        std::copy(element.begin(), element.begin() + components, out.begin() + i * components);
    }
    return out;
}

void read_material(const cgltf_material* source, Material& material) {
    if (!source) return;
    if (source->has_pbr_metallic_roughness) {
        const cgltf_pbr_metallic_roughness& pbr = source->pbr_metallic_roughness;
        std::copy(pbr.base_color_factor, pbr.base_color_factor + 4, material.baseColor);
        material.metallic = pbr.metallic_factor;
        material.roughness = pbr.roughness_factor;
    }
    std::copy(source->emissive_factor, source->emissive_factor + 3, material.emissive);
    material.doubleSided = source->double_sided;
}

}  // namespace

bool decode_glb(const uint8_t* data, size_t size, std::vector<Primitive>& out, std::string& error) {
    out.clear();
    cgltf_options options = {};
    cgltf_data* gltf = nullptr;
    if (cgltf_parse(&options, data, size, &gltf) != cgltf_result_success) {
        error = "not a glTF binary";
        return false;
    }
    struct Release {
        cgltf_data* gltf;
        ~Release() { cgltf_free(gltf); }
    } release{gltf};

    // A GLB keeps buffer 0 in its BIN chunk; buffers with a URI are not read.
    if (gltf->buffers_count > 0 && gltf->bin && !gltf->buffers[0].uri && gltf->bin_size >= gltf->buffers[0].size) {
        gltf->buffers[0].data = (void*)gltf->bin;
        gltf->buffers[0].data_free_method = cgltf_data_free_method_none;
    }
    if (cgltf_validate(gltf) != cgltf_result_success) {
        error = "glTF binary is inconsistent";
        return false;
    }
    if (!decode_meshopt_views(gltf, error)) return false;

    for (cgltf_size n = 0; n < gltf->nodes_count; n++) {
        const cgltf_node* node = &gltf->nodes[n];
        if (!node->mesh) continue;
        float m[16];
        cgltf_node_transform_world(node, m);

        for (cgltf_size p = 0; p < node->mesh->primitives_count; p++) {
            const cgltf_primitive& source = node->mesh->primitives[p];
            if (source.type != cgltf_primitive_type_points && source.type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* position = cgltf_find_accessor(&source, cgltf_attribute_type_position, 0);
            if (!position || !cgltf_buffer_view_data(position->buffer_view)) continue;

            Primitive primitive;
            primitive.mode = source.type == cgltf_primitive_type_points ? PrimitiveMode::Points : PrimitiveMode::Triangles;

            primitive.positions = unpack(position, 3);
            for (size_t i = 0; i < position->count; i++) {
                float* v = &primitive.positions[i * 3];
                float x = v[0], y = v[1], z = v[2];
                v[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
                v[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
                v[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
            }

            const cgltf_accessor* normal = cgltf_find_accessor(&source, cgltf_attribute_type_normal, 0);
            if (normal && normal->count == position->count && cgltf_buffer_view_data(normal->buffer_view)) {
                primitive.normals = unpack(normal, 3);
                for (size_t i = 0; i < normal->count; i++) {
                    float* v = &primitive.normals[i * 3];
                    float x = v[0], y = v[1], z = v[2];
                    float nx = m[0] * x + m[4] * y + m[8] * z;
                    float ny = m[1] * x + m[5] * y + m[9] * z;
                    float nz = m[2] * x + m[6] * y + m[10] * z;
                    float length = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (length > 0.0f) {
                        nx /= length;
                        ny /= length;
                        nz /= length;
                    }
                    v[0] = nx;
                    v[1] = ny;
                    v[2] = nz;
                }
            }

            const cgltf_accessor* color = cgltf_find_accessor(&source, cgltf_attribute_type_color, 0);
            if (color && color->count == position->count && cgltf_buffer_view_data(color->buffer_view)) {
                size_t components = cgltf_num_components(color->type) == 4 ? 4 : 3;
                std::vector<float> values = unpack(color, components);
                primitive.colors.resize(color->count * 4);
                rgba8_from_floats(values.data(), color->count, components, primitive.colors.data());
            }

            if (source.indices && cgltf_buffer_view_data(source.indices->buffer_view)) {
                primitive.indices.resize(source.indices->count);
                for (cgltf_size i = 0; i < source.indices->count; i++) {
                    primitive.indices[i] = (uint32_t)cgltf_accessor_read_index(source.indices, i);
                }
            } else if (primitive.mode == PrimitiveMode::Triangles) {
                primitive.indices.resize(position->count);
                for (size_t i = 0; i < position->count; i++) primitive.indices[i] = (uint32_t)i;
            }

            read_material(source.material, primitive.material);
            out.push_back(std::move(primitive));
        }
    }
    return true;
}

}  // namespace glbcore
