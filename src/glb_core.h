#pragma once
//
// glb_core.h - Runtime-agnostic GLB assembly core.
//
// Single source of truth for GLB generation, shared by the Node (N-API) and
// Python (pybind11) bindings. Contains NO node_api.h / Python.h: just plain
// C++ on raw buffers. All functions are `inline` and all file-scope state is
// `static` so the header can be included in multiple translation units.
//
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SA_X86 1
#else
#define SA_X86 0
#endif

#define SA_FORCE_INLINE inline __attribute__((always_inline))
#define SA_RESTRICT __restrict
#define SA_ALIGN_BYTES 32
#if (defined(__GNUC__) || defined(__clang__)) && SA_X86
#define SA_AVX2_BMI2_TARGET __attribute__((target("avx2,bmi2")))
#else
#define SA_AVX2_BMI2_TARGET
#endif

namespace glbcore {

static inline void* aligned_malloc(size_t size) {
#if SA_X86
    void* ptr = _mm_malloc(size, SA_ALIGN_BYTES);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, SA_ALIGN_BYTES, size) != 0) ptr = nullptr;
#endif
    if (!ptr) abort();
    return ptr;
}

static inline void aligned_free(void* ptr) {
#if SA_X86
    _mm_free(ptr);
#else
    free(ptr);
#endif
}

static inline unsigned int worker_thread_count(unsigned int fallback = 4) {
    unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? fallback : n;
}

static inline bool has_fast_cpu_features() {
#if SA_X86 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
}

// Type colors (SoA), 8 structure types.
static const float TYPE_COLORS_R[8] = {0.5f, 1.0f, 0.267f, 0.267f, 1.0f, 1.0f, 0.267f, 0.6f};
static const float TYPE_COLORS_G[8] = {0.5f, 0.267f, 1.0f, 0.267f, 1.0f, 0.267f, 1.0f, 0.6f};
static const float TYPE_COLORS_B[8] = {0.5f, 0.267f, 0.267f, 1.0f, 0.267f, 1.0f, 1.0f, 0.6f};

SA_FORCE_INLINE uint32_t expand_bits_3d(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

SA_FORCE_INLINE uint32_t morton3d_scalar(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits_3d(x) | (expand_bits_3d(y) << 1) | (expand_bits_3d(z) << 2);
}

#if SA_X86
SA_AVX2_BMI2_TARGET static inline uint32_t morton3d_bmi2(uint32_t x, uint32_t y, uint32_t z) {
    return _pdep_u32(x, 0x92492492) | _pdep_u32(y, 0x24924924) | _pdep_u32(z, 0x49249249);
}
#endif

struct RadixHist {
    uint32_t count[256];
};

static inline void radix_count_chunk(const uint32_t* SA_RESTRICT keys, size_t start, size_t end, int shift, RadixHist* hist) {
    memset(hist->count, 0, sizeof(hist->count));
    for (size_t i = start; i < end; i++) {
        hist->count[(keys[i] >> shift) & 0xFF]++;
    }
}

static inline void radix_scatter(
    const uint32_t* SA_RESTRICT srcKeys, const uint32_t* SA_RESTRICT srcIndices,
    uint32_t* SA_RESTRICT dstKeys, uint32_t* SA_RESTRICT dstIndices,
    size_t start, size_t end, int shift, uint32_t* localOffsets) {
    for (size_t i = start; i < end; i++) {
        uint8_t bucket = (srcKeys[i] >> shift) & 0xFF;
        uint32_t d = localOffsets[bucket]++;
        dstKeys[d] = srcKeys[i];
        dstIndices[d] = srcIndices[i];
    }
}

static inline void lock_free_radix_sort(uint32_t*& keys, uint32_t*& indices, size_t n, unsigned int numThreads) {
    uint32_t* tmpKeys = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    uint32_t* tmpIndices = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    uint32_t *srcK = keys, *dstK = tmpKeys;
    uint32_t *srcI = indices, *dstI = tmpIndices;

    std::vector<RadixHist> hists(numThreads);
    std::vector<std::thread> threads;
    size_t blockSize = (n + numThreads - 1) / numThreads;
    std::vector<std::vector<uint32_t>> threadOffsets(numThreads, std::vector<uint32_t>(256));

    for (int shift = 0; shift < 32; shift += 8) {
        threads.clear();
        for (unsigned int t = 0; t < numThreads; t++) {
            size_t start = t * blockSize, end = std::min(start + blockSize, n);
            if (start < n) threads.emplace_back(radix_count_chunk, srcK, start, end, shift, &hists[t]);
        }
        for (auto& th : threads) th.join();

        uint32_t globalOffsets[256];
        uint32_t running = 0;
        for (int b = 0; b < 256; b++) {
            globalOffsets[b] = running;
            for (unsigned int t = 0; t < numThreads; t++) running += hists[t].count[b];
        }
        for (int b = 0; b < 256; b++) {
            uint32_t offset = globalOffsets[b];
            for (unsigned int t = 0; t < numThreads; t++) {
                threadOffsets[t][b] = offset;
                offset += hists[t].count[b];
            }
        }

        threads.clear();
        for (unsigned int t = 0; t < numThreads; t++) {
            size_t start = t * blockSize, end = std::min(start + blockSize, n);
            if (start < n) threads.emplace_back(radix_scatter, srcK, srcI, dstK, dstI, start, end, shift, threadOffsets[t].data());
        }
        for (auto& th : threads) th.join();
        std::swap(srcK, dstK);
        std::swap(srcI, dstI);
    }

    if (srcK != keys) {
        memcpy(keys, srcK, n * sizeof(uint32_t));
        memcpy(indices, srcI, n * sizeof(uint32_t));
    }
    aligned_free(tmpKeys);
    aligned_free(tmpIndices);
}

static inline void colorize_by_type(
    const uint32_t* SA_RESTRICT indices, const uint16_t* SA_RESTRICT srcTypes,
    float* SA_RESTRICT dstColors, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        uint16_t t = srcTypes[indices[i]];
        uint32_t c = (t <= 7) ? t : 7;
        size_t out = i * 3;
        dstColors[out] = TYPE_COLORS_R[c];
        dstColors[out + 1] = TYPE_COLORS_G[c];
        dstColors[out + 2] = TYPE_COLORS_B[c];
    }
}

static inline void gather_positions(
    const uint32_t* SA_RESTRICT indices, const float* SA_RESTRICT srcPos,
    float* SA_RESTRICT dstPos, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        size_t s = indices[i] * 3, d = i * 3;
        dstPos[d] = srcPos[s];
        dstPos[d + 1] = srcPos[s + 1];
        dstPos[d + 2] = srcPos[s + 2];
    }
}

// Assemble a GLB container from a JSON chunk and a contiguous BIN blob.
static inline std::vector<uint8_t> assemble_glb(const char* json, size_t jsonLen, const uint8_t* bin, size_t binLen) {
    size_t jsonPad = (4 - (jsonLen % 4)) % 4;
    size_t binPad = (4 - (binLen % 4)) % 4;
    size_t jsonChunk = jsonLen + jsonPad;
    size_t binChunk = binLen + binPad;
    size_t total = 12 + 8 + jsonChunk + 8 + binChunk;

    std::vector<uint8_t> glb(total);
    uint8_t* p = glb.data();
    auto put = [&](uint32_t v) { memcpy(p, &v, 4); p += 4; };
    put(0x46546C67); put(2); put((uint32_t)total);          // GLB header
    put((uint32_t)jsonChunk); put(0x4E4F534A);              // JSON chunk header
    memcpy(p, json, jsonLen); p += jsonLen;
    memset(p, ' ', jsonPad); p += jsonPad;
    put((uint32_t)binChunk); put(0x004E4942);               // BIN chunk header
    if (binLen) { memcpy(p, bin, binLen); p += binLen; }
    memset(p, 0, binPad);
    return glb;
}

// ---- Entry points ---------------------------------------------------------

// Atom cloud: Morton-sort by position, colorize by structure type, emit GLB.
static inline std::vector<uint8_t> generate_atom_glb(
    const float* srcPos, size_t n, const uint16_t* srcTypes,
    const double minA[3], const double maxA[3]) {
    if (n == 0) return assemble_glb("{}", 2, nullptr, 0);

    const bool fast = has_fast_cpu_features();
    float* outPos = (float*)aligned_malloc(n * 3 * sizeof(float));
    float* outCol = (float*)aligned_malloc(n * 3 * sizeof(float));
    uint32_t* keys = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    uint32_t* indices = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));

    unsigned int numThreads = (n < 100000) ? 1 : worker_thread_count();
    size_t blockSize = (n + numThreads - 1) / numThreads;

    float invX = 1.0f / std::max(1e-10f, (float)(maxA[0] - minA[0]));
    float invY = 1.0f / std::max(1e-10f, (float)(maxA[1] - minA[1]));
    float invZ = 1.0f / std::max(1e-10f, (float)(maxA[2] - minA[2]));

    auto mortonWorker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            size_t p = i * 3;
            float x = (srcPos[p] - (float)minA[0]) * invX;
            float y = (srcPos[p + 1] - (float)minA[1]) * invY;
            float z = (srcPos[p + 2] - (float)minA[2]) * invZ;
            uint32_t ux = std::min(1023u, std::max(0u, (uint32_t)(x * 1023.0f)));
            uint32_t uy = std::min(1023u, std::max(0u, (uint32_t)(y * 1023.0f)));
            uint32_t uz = std::min(1023u, std::max(0u, (uint32_t)(z * 1023.0f)));
#if SA_X86
            keys[i] = fast ? morton3d_bmi2(ux, uy, uz) : morton3d_scalar(ux, uy, uz);
#else
            keys[i] = morton3d_scalar(ux, uy, uz);
#endif
            indices[i] = (uint32_t)i;
        }
    };

    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize, end = std::min(start + blockSize, n);
        if (start < n) threads.emplace_back(mortonWorker, start, end);
    }
    for (auto& th : threads) th.join();

    lock_free_radix_sort(keys, indices, n, numThreads);

    threads.clear();
    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize, end = std::min(start + blockSize, n);
        if (start < n) threads.emplace_back([&, start, end]() {
            gather_positions(indices, srcPos, outPos, start, end);
            colorize_by_type(indices, srcTypes, outCol, start, end);
        });
    }
    for (auto& th : threads) th.join();

    size_t posBytes = n * 3 * sizeof(float);
    char json[2048];
    int jsonLen = snprintf(json, sizeof(json),
        R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Atoms"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"mode":0}],"name":"AtomCloud"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962}],"buffers":[{"byteLength":%zu}]})",
        n, minA[0], minA[1], minA[2], maxA[0], maxA[1], maxA[2], n, posBytes, posBytes, posBytes, posBytes * 2);

    std::vector<uint8_t> bin(posBytes * 2);
    memcpy(bin.data(), outPos, posBytes);
    memcpy(bin.data() + posBytes, outCol, posBytes);

    aligned_free(outPos);
    aligned_free(outCol);
    aligned_free(keys);
    aligned_free(indices);
    return assemble_glb(json, jsonLen, bin.data(), bin.size());
}

// Pre-colored point cloud (colors VEC3 or VEC4).
static inline std::vector<uint8_t> generate_point_cloud_glb(
    const float* positions, size_t posCount, const float* colors, size_t colCount,
    const double minA[3], const double maxA[3]) {
    size_t atomCount = posCount / 3;
    bool isVec4 = (colCount == atomCount * 4);
    size_t posBytes = posCount * sizeof(float);
    size_t colBytes = colCount * sizeof(float);

    char json[2048];
    int jsonLen = snprintf(json, sizeof(json),
        R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Atoms"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"mode":0}],"name":"AtomCloud"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"%s"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962}],"buffers":[{"byteLength":%zu}]})",
        atomCount, minA[0], minA[1], minA[2], maxA[0], maxA[1], maxA[2],
        atomCount, isVec4 ? "VEC4" : "VEC3", posBytes, posBytes, colBytes, posBytes + colBytes);

    std::vector<uint8_t> bin(posBytes + colBytes);
    memcpy(bin.data(), positions, posBytes);
    memcpy(bin.data() + posBytes, colors, colBytes);
    return assemble_glb(json, jsonLen, bin.data(), bin.size());
}

struct Material {
    double baseColor[4] = {1, 1, 1, 1};
    double metallic = 0;
    double roughness = 1;
    double emissive[3] = {0, 0, 0};
    bool doubleSided = true;
};

// Indexed triangle mesh with normals, optional VEC4 colors, and a PBR material.
// Indices are uint32 (SCALAR componentType 5125).
static inline std::vector<uint8_t> generate_mesh_glb(
    const float* positions, size_t posCount, const float* normals, size_t normCount,
    const uint32_t* indices, size_t idxCount, const float* colors, size_t colCount,
    const double bounds[6], const Material& mat) {
    size_t vertexCount = posCount / 3;
    bool hasColors = (colors != nullptr && colCount > 0);

    size_t posBytes = posCount * sizeof(float);
    size_t normBytes = normCount * sizeof(float);
    size_t colBytes = hasColors ? colCount * sizeof(float) : 0;
    size_t idxBytes = idxCount * sizeof(uint32_t);
    size_t binTotal = posBytes + normBytes + colBytes + idxBytes;

    const char* ds = mat.doubleSided ? "true" : "false";
    char json[8192];
    int jsonLen;
    if (hasColors) {
        jsonLen = snprintf(json, sizeof(json),
            R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Mesh"}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[%.4f,%.4f,%.4f,%.4f],"metallicFactor":%.4f,"roughnessFactor":%.4f},"emissiveFactor":[%.4f,%.4f,%.4f],"doubleSided":%s}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"COLOR_0":2},"indices":3,"material":0,"mode":4}],"name":"MeshGeometry"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"},{"bufferView":2,"componentType":5126,"count":%zu,"type":"VEC4"},{"bufferView":3,"componentType":5125,"count":%zu,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34963}],"buffers":[{"byteLength":%zu}]})",
            mat.baseColor[0], mat.baseColor[1], mat.baseColor[2], mat.baseColor[3], mat.metallic, mat.roughness,
            mat.emissive[0], mat.emissive[1], mat.emissive[2], ds,
            vertexCount, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
            vertexCount, vertexCount, idxCount,
            posBytes, posBytes, normBytes, posBytes + normBytes, colBytes, posBytes + normBytes + colBytes, idxBytes, binTotal);
    } else {
        jsonLen = snprintf(json, sizeof(json),
            R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Mesh"}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[%.4f,%.4f,%.4f,%.4f],"metallicFactor":%.4f,"roughnessFactor":%.4f},"emissiveFactor":[%.4f,%.4f,%.4f],"doubleSided":%s}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2,"material":0,"mode":4}],"name":"MeshGeometry"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"},{"bufferView":2,"componentType":5125,"count":%zu,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34963}],"buffers":[{"byteLength":%zu}]})",
            mat.baseColor[0], mat.baseColor[1], mat.baseColor[2], mat.baseColor[3], mat.metallic, mat.roughness,
            mat.emissive[0], mat.emissive[1], mat.emissive[2], ds,
            vertexCount, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
            vertexCount, idxCount,
            posBytes, posBytes, normBytes, posBytes + normBytes, idxBytes, binTotal);
    }

    std::vector<uint8_t> bin(binTotal);
    uint8_t* p = bin.data();
    memcpy(p, positions, posBytes); p += posBytes;
    memcpy(p, normals, normBytes); p += normBytes;
    if (hasColors) { memcpy(p, colors, colBytes); p += colBytes; }
    memcpy(p, indices, idxBytes);
    return assemble_glb(json, jsonLen, bin.data(), bin.size());
}

}  // namespace glbcore
