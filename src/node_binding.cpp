// node_binding.cpp - N-API surface of the shared core (glb_core.h, colors.h, smoothing.h).
#include <node_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "colors.h"
#include "glb_core.h"
#include "smoothing.h"

namespace {

struct Typed {
    napi_typedarray_type type = napi_uint8_array;
    size_t length = 0;
    void* data = nullptr;
};

bool is_nullish(napi_env env, napi_value value) {
    napi_valuetype type;
    napi_typeof(env, value, &type);
    return type == napi_undefined || type == napi_null;
}

bool read_typed(napi_env env, napi_value value, Typed& out) {
    bool isTyped = false;
    napi_is_typedarray(env, value, &isTyped);
    if (!isTyped) return false;
    napi_get_typedarray_info(env, value, &out.type, &out.length, &out.data, nullptr, nullptr);
    return true;
}

napi_value fail(napi_env env, const char* message) {
    napi_throw_type_error(env, nullptr, message);
    return nullptr;
}

napi_value to_buffer(napi_env env, const std::vector<uint8_t>& bytes) {
    napi_value result;
    void* data;
    napi_create_buffer_copy(env, bytes.size(), bytes.data(), &data, &result);
    return result;
}

template <typename T>
napi_value to_typed(napi_env env, const std::vector<T>& values, napi_typedarray_type type) {
    napi_value buffer, array;
    void* data;
    napi_create_arraybuffer(env, values.size() * sizeof(T), &data, &buffer);
    if (!values.empty()) std::memcpy(data, values.data(), values.size() * sizeof(T));
    napi_create_typedarray(env, type, values.size(), buffer, 0, &array);
    return array;
}

// Colours as RGBA8 (Uint8Array, 4 per item) or floats in [0, 1] (Float32Array, 3 or 4 per item).
bool read_colors(napi_env env, napi_value value, size_t count, std::vector<uint8_t>& rgba, bool& present) {
    present = false;
    if (is_nullish(env, value)) return true;
    Typed colors;
    if (!read_typed(env, value, colors)) return false;
    if (colors.type == napi_uint8_array && colors.length == count * 4) {
        rgba.assign((uint8_t*)colors.data, (uint8_t*)colors.data + count * 4);
    } else if (colors.type == napi_float32_array && (colors.length == count * 3 || colors.length == count * 4)) {
        rgba.resize(count * 4);
        glbcore::rgba8_from_floats((const float*)colors.data, count, colors.length / (count ? count : 1), rgba.data());
    } else {
        return false;
    }
    present = true;
    return true;
}

double read_number(napi_env env, napi_value object, const char* key, double fallback) {
    bool has = false;
    napi_has_named_property(env, object, key, &has);
    if (!has) return fallback;
    napi_value value;
    napi_get_named_property(env, object, key, &value);
    double out = fallback;
    if (napi_get_value_double(env, value, &out) != napi_ok) return fallback;
    return out;
}

void read_numbers(napi_env env, napi_value object, const char* key, float* out, uint32_t count) {
    bool has = false;
    napi_has_named_property(env, object, key, &has);
    if (!has) return;
    napi_value array;
    napi_get_named_property(env, object, key, &array);
    for (uint32_t i = 0; i < count; i++) {
        napi_value item;
        double value;
        if (napi_get_element(env, array, i, &item) == napi_ok && napi_get_value_double(env, item, &value) == napi_ok) {
            out[i] = (float)value;
        }
    }
}

glbcore::Material read_material(napi_env env, napi_value object) {
    glbcore::Material material;
    if (is_nullish(env, object)) return material;
    read_numbers(env, object, "baseColor", material.baseColor, 4);
    read_numbers(env, object, "emissive", material.emissive, 3);
    material.metallic = (float)read_number(env, object, "metallic", material.metallic);
    material.roughness = (float)read_number(env, object, "roughness", material.roughness);
    bool has = false;
    napi_has_named_property(env, object, "doubleSided", &has);
    if (has) {
        napi_value value;
        napi_get_named_property(env, object, "doubleSided", &value);
        napi_get_value_bool(env, value, &material.doubleSided);
    }
    return material;
}

napi_value material_object(napi_env env, const glbcore::Material& material) {
    napi_value object, array, value;
    napi_create_object(env, &object);
    napi_create_array_with_length(env, 4, &array);
    for (uint32_t i = 0; i < 4; i++) {
        napi_create_double(env, material.baseColor[i], &value);
        napi_set_element(env, array, i, value);
    }
    napi_set_named_property(env, object, "baseColor", array);
    napi_create_array_with_length(env, 3, &array);
    for (uint32_t i = 0; i < 3; i++) {
        napi_create_double(env, material.emissive[i], &value);
        napi_set_element(env, array, i, value);
    }
    napi_set_named_property(env, object, "emissive", array);
    napi_create_double(env, material.metallic, &value);
    napi_set_named_property(env, object, "metallic", value);
    napi_create_double(env, material.roughness, &value);
    napi_set_named_property(env, object, "roughness", value);
    napi_get_boolean(env, material.doubleSided, &value);
    napi_set_named_property(env, object, "doubleSided", value);
    return object;
}

// encodePointCloud(positions: Float32Array, colors?: Uint8Array | Float32Array | null): Buffer
napi_value EncodePointCloud(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Typed positions;
    if (argc < 1 || !read_typed(env, args[0], positions) || positions.type != napi_float32_array || positions.length % 3 != 0) {
        return fail(env, "encodePointCloud: positions must be a Float32Array of xyz triples");
    }
    const size_t count = positions.length / 3;
    std::vector<uint8_t> rgba;
    bool hasColors = false;
    if (argc > 1 && !read_colors(env, args[1], count, rgba, hasColors)) {
        return fail(env, "encodePointCloud: colors must be RGBA8 (Uint8Array) or RGB/RGBA floats (Float32Array), one per point");
    }
    return to_buffer(env, glbcore::encode_points((const float*)positions.data, count, hasColors ? rgba.data() : nullptr));
}

// encodeMesh(positions, normals, indices, colors?, material?): Buffer
napi_value EncodeMesh(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Typed positions, normals, indices;
    if (argc < 3 || !read_typed(env, args[0], positions) || positions.type != napi_float32_array || positions.length % 3 != 0) {
        return fail(env, "encodeMesh: positions must be a Float32Array of xyz triples");
    }
    const size_t vertexCount = positions.length / 3;
    if (!read_typed(env, args[1], normals) || normals.type != napi_float32_array || normals.length != positions.length) {
        return fail(env, "encodeMesh: normals must be a Float32Array matching positions");
    }
    if (!read_typed(env, args[2], indices) || (indices.type != napi_uint32_array && indices.type != napi_uint16_array)) {
        return fail(env, "encodeMesh: indices must be a Uint32Array or Uint16Array");
    }
    std::vector<uint32_t> wide(indices.length);
    for (size_t i = 0; i < indices.length; i++) {
        wide[i] = indices.type == napi_uint32_array ? ((const uint32_t*)indices.data)[i] : ((const uint16_t*)indices.data)[i];
    }
    std::vector<uint8_t> rgba;
    bool hasColors = false;
    if (argc > 3 && !read_colors(env, args[3], vertexCount, rgba, hasColors)) {
        return fail(env, "encodeMesh: colors must be RGBA8 (Uint8Array) or RGB/RGBA floats (Float32Array), one per vertex");
    }
    glbcore::Material material = argc > 4 ? read_material(env, args[4]) : glbcore::Material();
    return to_buffer(env, glbcore::encode_mesh(
        (const float*)positions.data, (const float*)normals.data, vertexCount,
        hasColors ? rgba.data() : nullptr, wide.data(), wide.size(), material));
}

// decodeGLB(bytes: Uint8Array): Array<{ mode, positions, normals, colors, indices, material }>
napi_value DecodeGLB(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Typed bytes;
    if (argc < 1 || !read_typed(env, args[0], bytes) || bytes.type != napi_uint8_array) {
        return fail(env, "decodeGLB: expected a Uint8Array or Buffer");
    }
    std::vector<glbcore::Primitive> primitives;
    std::string error;
    if (!glbcore::decode_glb((const uint8_t*)bytes.data, bytes.length, primitives, error)) {
        napi_throw_error(env, nullptr, ("decodeGLB: " + error).c_str());
        return nullptr;
    }

    napi_value result, value, nullValue;
    napi_get_null(env, &nullValue);
    napi_create_array_with_length(env, primitives.size(), &result);
    for (size_t i = 0; i < primitives.size(); i++) {
        const glbcore::Primitive& primitive = primitives[i];
        napi_value object;
        napi_create_object(env, &object);
        napi_create_string_utf8(env, primitive.mode == glbcore::PrimitiveMode::Points ? "points" : "triangles", NAPI_AUTO_LENGTH, &value);
        napi_set_named_property(env, object, "mode", value);
        napi_set_named_property(env, object, "positions", to_typed(env, primitive.positions, napi_float32_array));
        napi_set_named_property(env, object, "normals", primitive.normals.empty() ? nullValue : to_typed(env, primitive.normals, napi_float32_array));
        napi_set_named_property(env, object, "colors", primitive.colors.empty() ? nullValue : to_typed(env, primitive.colors, napi_uint8_array));
        napi_set_named_property(env, object, "indices", primitive.mode == glbcore::PrimitiveMode::Points ? nullValue : to_typed(env, primitive.indices, napi_uint32_array));
        napi_set_named_property(env, object, "material", material_object(env, primitive.material));
        napi_set_element(env, result, (uint32_t)i, object);
    }
    return result;
}

// colorsByType(types: Uint16Array): Uint8Array (RGBA8)
napi_value ColorsByType(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Typed types;
    if (argc < 1 || !read_typed(env, args[0], types) || types.type != napi_uint16_array) {
        return fail(env, "colorsByType: types must be a Uint16Array");
    }
    std::vector<uint8_t> rgba(types.length * 4);
    glbcore::colors_by_type((const uint16_t*)types.data, types.length, rgba.data());
    return to_typed(env, rgba, napi_uint8_array);
}

// propertyColors(values: Float32Array, min: number, max: number, gradient: number): Uint8Array (RGBA8)
napi_value PropertyColors(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Typed values;
    if (argc < 4 || !read_typed(env, args[0], values) || values.type != napi_float32_array) {
        return fail(env, "propertyColors: values must be a Float32Array");
    }
    double minValue = 0, maxValue = 0;
    int32_t gradient = 0;
    napi_get_value_double(env, args[1], &minValue);
    napi_get_value_double(env, args[2], &maxValue);
    napi_get_value_int32(env, args[3], &gradient);
    std::vector<uint8_t> rgba(values.length * 4);
    glbcore::property_colors((const float*)values.data, values.length, minValue, maxValue, gradient, rgba.data());
    return to_typed(env, rgba, napi_uint8_array);
}

// taubinSmooth(positions: Float32Array, indices: Uint32Array, iterations: number): void (in place)
napi_value TaubinSmooth(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    Typed positions, indices;
    if (argc < 3 || !read_typed(env, args[0], positions) || positions.type != napi_float32_array
        || !read_typed(env, args[1], indices) || indices.type != napi_uint32_array) {
        return fail(env, "taubinSmooth: expected (Float32Array positions, Uint32Array indices, iterations)");
    }
    int32_t iterations = 0;
    napi_get_value_int32(env, args[2], &iterations);
    glbcore::taubin_smooth((float*)positions.data, positions.length / 3, (const uint32_t*)indices.data, indices.length, iterations);
    return nullptr;
}

void define(napi_env env, napi_value exports, const char* name, napi_callback callback) {
    napi_value fn;
    napi_create_function(env, name, NAPI_AUTO_LENGTH, callback, nullptr, &fn);
    napi_set_named_property(env, exports, name, fn);
}

napi_value Init(napi_env env, napi_value exports) {
    define(env, exports, "encodePointCloud", EncodePointCloud);
    define(env, exports, "encodeMesh", EncodeMesh);
    define(env, exports, "decodeGLB", DecodeGLB);
    define(env, exports, "colorsByType", ColorsByType);
    define(env, exports, "propertyColors", PropertyColors);
    define(env, exports, "taubinSmooth", TaubinSmooth);
    napi_value step;
    napi_create_double(env, glbcore::POSITION_STEP, &step);
    napi_set_named_property(env, exports, "POSITION_STEP", step);
    return exports;
}

}  // namespace

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
