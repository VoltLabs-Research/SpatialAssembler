#include <node_api.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>
#include <immintrin.h>
#include <mutex>
#include <cstdio>
#include <string>

// CONFIGURATION AND PERFORMANCE MACROS
#define FORCE_INLINE inline __attribute__((always_inline))
#define RESTRICT __restrict
#define ALIGN_BYTES 32
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#if defined(__GNUC__) || defined(__clang__)
#define SPATIAL_ASSEMBLER_AVX2_BMI2_TARGET __attribute__((target("avx2,bmi2")))
#else
#define SPATIAL_ASSEMBLER_AVX2_BMI2_TARGET
#endif

void* aligned_malloc(size_t size) {
    void* ptr = _mm_malloc(size, ALIGN_BYTES);
    if (!ptr) abort();
    return ptr;
}

void aligned_free(void* ptr) {
    _mm_free(ptr);
}

FORCE_INLINE unsigned int getWorkerThreadCount(unsigned int fallback = 1) {
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) {
        return fallback;
    }

    return numThreads;
}

bool hasSpatialAssemblerFastCpuFeatures() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
#else
    return false;
#endif
}

// COLOR TABLES - SOA Layout for SIMD access
// Type colors: 8 types, RGB stored as separate rows for SIMD loading
static const float TYPE_COLORS_R[] __attribute__((aligned(32))) = {
    0.5f, 1.0f, 0.267f, 0.267f, 1.0f, 1.0f, 0.267f, 0.6f
};
static const float TYPE_COLORS_G[] __attribute__((aligned(32))) = {
    0.5f, 0.267f, 1.0f, 0.267f, 1.0f, 0.267f, 1.0f, 0.6f
};
static const float TYPE_COLORS_B[] __attribute__((aligned(32))) = {
    0.5f, 0.267f, 0.267f, 1.0f, 0.267f, 1.0f, 1.0f, 0.6f
};

// GRADIENT LUT - 1024 entries for smooth color transitions
#define GRADIENT_LUT_SIZE 1024
#define GRADIENT_TYPE_COUNT 10
static float GRADIENT_LUT[3 * GRADIENT_LUT_SIZE * GRADIENT_TYPE_COUNT] __attribute__((aligned(32)));
static bool GRADIENT_LUT_INIT = false;

FORCE_INLINE void lerp3(float* out, const float* c0, const float* c1, float t) {
    out[0] = c0[0] + (c1[0] - c0[0]) * t;
    out[1] = c0[1] + (c1[1] - c0[1]) * t;
    out[2] = c0[2] + (c1[2] - c0[2]) * t;
}

// Gradient codes are a wire contract with the daemon (FilterEvaluator's
// GradientCode): 0 Viridis, 1 Plasma, 2 BlueRed, 3 Grayscale, 4 Magma,
// 5 Inferno, 6 Cividis, 7 RdBu, 8 Coolwarm, 9 Jet. Codes 0-3 predate this
// table and must keep their meaning. Stop colors mirror the client gradient
// previews (GradientPreview GRADIENT_CSS) so baked colors match the picker.
struct GradientStop {
    float t;
    float rgb[3];
};

static const GradientStop VIRIDIS_STOPS[] = {
    {0.000f, {0.2667f, 0.0039f, 0.3294f}}, {0.111f, {0.2824f, 0.1569f, 0.4706f}},
    {0.222f, {0.2431f, 0.2902f, 0.5373f}}, {0.333f, {0.1922f, 0.4078f, 0.5569f}},
    {0.444f, {0.1490f, 0.5137f, 0.5608f}}, {0.556f, {0.1216f, 0.6196f, 0.5373f}},
    {0.667f, {0.2078f, 0.7176f, 0.4745f}}, {0.778f, {0.4314f, 0.8078f, 0.3451f}},
    {0.889f, {0.7098f, 0.8706f, 0.1686f}}, {1.000f, {0.9922f, 0.9059f, 0.1451f}}
};

static const GradientStop PLASMA_STOPS[] = {
    {0.000f, {0.0510f, 0.0314f, 0.5294f}}, {0.111f, {0.2745f, 0.0118f, 0.6235f}},
    {0.222f, {0.4471f, 0.0039f, 0.6588f}}, {0.333f, {0.6118f, 0.0902f, 0.6196f}},
    {0.444f, {0.7412f, 0.2157f, 0.5255f}}, {0.556f, {0.8471f, 0.3412f, 0.4196f}},
    {0.667f, {0.9294f, 0.4745f, 0.3255f}}, {0.778f, {0.9843f, 0.6235f, 0.2275f}},
    {0.889f, {0.9922f, 0.7922f, 0.1490f}}, {1.000f, {0.9412f, 0.9765f, 0.1294f}}
};

static const GradientStop BLUERED_STOPS[] = {
    {0.0f, {0.0f, 0.0f, 1.0f}}, {0.5f, {1.0f, 1.0f, 1.0f}}, {1.0f, {1.0f, 0.0f, 0.0f}}
};

static const GradientStop GRAYSCALE_STOPS[] = {
    {0.0f, {0.0f, 0.0f, 0.0f}}, {1.0f, {1.0f, 1.0f, 1.0f}}
};

static const GradientStop MAGMA_STOPS[] = {
    {0.000f, {0.0000f, 0.0000f, 0.0157f}}, {0.125f, {0.1098f, 0.0627f, 0.2667f}},
    {0.250f, {0.3176f, 0.0706f, 0.4863f}}, {0.375f, {0.5098f, 0.1490f, 0.5059f}},
    {0.500f, {0.7176f, 0.2157f, 0.4745f}}, {0.625f, {0.9059f, 0.3216f, 0.3882f}},
    {0.750f, {0.9843f, 0.5294f, 0.3804f}}, {0.875f, {0.9961f, 0.7608f, 0.5294f}},
    {1.000f, {0.9882f, 0.9922f, 0.7490f}}
};

static const GradientStop INFERNO_STOPS[] = {
    {0.000f, {0.0000f, 0.0000f, 0.0157f}}, {0.125f, {0.1059f, 0.0471f, 0.2588f}},
    {0.250f, {0.2902f, 0.0471f, 0.4196f}}, {0.375f, {0.4706f, 0.1098f, 0.4275f}},
    {0.500f, {0.6471f, 0.1725f, 0.3765f}}, {0.625f, {0.8118f, 0.2667f, 0.2745f}},
    {0.750f, {0.9294f, 0.4118f, 0.1451f}}, {0.875f, {0.9843f, 0.6039f, 0.0235f}},
    {1.000f, {0.9882f, 1.0000f, 0.6431f}}
};

static const GradientStop CIVIDIS_STOPS[] = {
    {0.000f, {0.0000f, 0.1255f, 0.3176f}}, {0.125f, {0.0706f, 0.2314f, 0.4000f}},
    {0.250f, {0.2000f, 0.2745f, 0.4275f}}, {0.375f, {0.3137f, 0.3490f, 0.4706f}},
    {0.500f, {0.4196f, 0.4235f, 0.4941f}}, {0.625f, {0.5529f, 0.4941f, 0.4863f}},
    {0.750f, {0.7255f, 0.5686f, 0.4588f}}, {0.875f, {0.8667f, 0.6902f, 0.3882f}},
    {1.000f, {0.9922f, 0.9176f, 0.2706f}}
};

static const GradientStop RDBU_STOPS[] = {
    {0.0f, {0.4039f, 0.0000f, 0.1216f}}, {0.1f, {0.6980f, 0.0941f, 0.1686f}},
    {0.2f, {0.8392f, 0.3765f, 0.3020f}}, {0.3f, {0.9569f, 0.6471f, 0.5098f}},
    {0.4f, {0.9922f, 0.8588f, 0.7804f}}, {0.5f, {0.9686f, 0.9686f, 0.9686f}},
    {0.6f, {0.8196f, 0.8980f, 0.9412f}}, {0.7f, {0.5725f, 0.7725f, 0.8706f}},
    {0.8f, {0.2627f, 0.5765f, 0.7647f}}, {0.9f, {0.1294f, 0.4000f, 0.6745f}},
    {1.0f, {0.0196f, 0.1882f, 0.3804f}}
};

static const GradientStop COOLWARM_STOPS[] = {
    {0.000f, {0.2314f, 0.2980f, 0.7529f}}, {0.125f, {0.3686f, 0.4902f, 0.9059f}},
    {0.250f, {0.5333f, 0.6667f, 0.9412f}}, {0.375f, {0.7098f, 0.8078f, 0.9529f}},
    {0.500f, {0.8667f, 0.8980f, 0.9255f}}, {0.625f, {0.9529f, 0.8275f, 0.7569f}},
    {0.750f, {0.9412f, 0.6627f, 0.4863f}}, {0.875f, {0.8588f, 0.4235f, 0.3176f}},
    {1.000f, {0.7059f, 0.0157f, 0.1608f}}
};

static const GradientStop JET_STOPS[] = {
    {0.000f, {0.0f, 0.0f, 0.5f}}, {0.125f, {0.0f, 0.0f, 1.0f}},
    {0.375f, {0.0f, 1.0f, 1.0f}}, {0.625f, {1.0f, 1.0f, 0.0f}},
    {0.875f, {1.0f, 0.0f, 0.0f}}, {1.000f, {0.5f, 0.0f, 0.0f}}
};

struct GradientDef {
    const GradientStop* stops;
    int count;
};

static const GradientDef GRADIENT_DEFS[GRADIENT_TYPE_COUNT] = {
    {VIRIDIS_STOPS, 10},
    {PLASMA_STOPS, 10},
    {BLUERED_STOPS, 3},
    {GRAYSCALE_STOPS, 2},
    {MAGMA_STOPS, 9},
    {INFERNO_STOPS, 9},
    {CIVIDIS_STOPS, 9},
    {RDBU_STOPS, 11},
    {COOLWARM_STOPS, 9},
    {JET_STOPS, 6}
};

void initGradientLUT() {
    if (GRADIENT_LUT_INIT) return;

    for (int g = 0; g < GRADIENT_TYPE_COUNT; g++) {
        const GradientDef& def = GRADIENT_DEFS[g];
        for (int i = 0; i < GRADIENT_LUT_SIZE; i++) {
            float t = i / (float)(GRADIENT_LUT_SIZE - 1);
            int hi = 1;
            while (hi < def.count - 1 && def.stops[hi].t < t) hi++;
            const GradientStop& s0 = def.stops[hi - 1];
            const GradientStop& s1 = def.stops[hi];
            float span = s1.t - s0.t;
            float lt = span > 0.0f ? (t - s0.t) / span : 0.0f;
            lt = std::max(0.0f, std::min(1.0f, lt));

            float rgb[3];
            lerp3(rgb, s0.rgb, s1.rgb, lt);

            int idx = (g * GRADIENT_LUT_SIZE * 3) + (i * 3);
            GRADIENT_LUT[idx] = rgb[0];
            GRADIENT_LUT[idx + 1] = rgb[1];
            GRADIENT_LUT[idx + 2] = rgb[2];
        }
    }
    GRADIENT_LUT_INIT = true;
}

FORCE_INLINE uint32_t expandBits3D(uint32_t value) {
    value &= 0x000003ffu;
    value = (value | (value << 16)) & 0x030000FFu;
    value = (value | (value << 8)) & 0x0300F00Fu;
    value = (value | (value << 4)) & 0x030C30C3u;
    value = (value | (value << 2)) & 0x09249249u;
    return value;
}

FORCE_INLINE uint32_t morton3DScalar(uint32_t x, uint32_t y, uint32_t z) {
    return expandBits3D(x) | (expandBits3D(y) << 1) | (expandBits3D(z) << 2);
}

// MORTON ENCODING (BMI2 HARDWARE)
SPATIAL_ASSEMBLER_AVX2_BMI2_TARGET uint32_t morton3D_BMI2(uint32_t x, uint32_t y, uint32_t z) {
    return _pdep_u32(x, 0x92492492) | 
           _pdep_u32(y, 0x24924924) | 
           _pdep_u32(z, 0x49249249);
}

// LOCK-FREE RADIX SORT - Per-thread local histograms, no atomics
struct RadixHist {
    uint32_t count[256];
};

void radixCountChunk(const uint32_t* RESTRICT keys, size_t start, size_t end, int shift, RadixHist* hist) {
    memset(hist->count, 0, sizeof(hist->count));
    for (size_t i = start; i < end; i += 4) {
        if (i + 3 < end) {
            hist->count[(keys[i] >> shift) & 0xFF]++;
            hist->count[(keys[i+1] >> shift) & 0xFF]++;
            hist->count[(keys[i+2] >> shift) & 0xFF]++;
            hist->count[(keys[i+3] >> shift) & 0xFF]++;
        } else {
            for (size_t j = i; j < end; j++) {
                hist->count[(keys[j] >> shift) & 0xFF]++;
            }
        }
    }
}

// Lock-free scatter using pre-computed per-thread offsets
void radixScatterLockFree(
    const uint32_t* RESTRICT srcKeys, const uint32_t* RESTRICT srcIndices,
    uint32_t* RESTRICT dstKeys, uint32_t* RESTRICT dstIndices,
    size_t start, size_t end, int shift,
    uint32_t* localOffsets 
) {
    for (size_t i = start; i < end; i++) {
        uint8_t bucket = (srcKeys[i] >> shift) & 0xFF;
        uint32_t destIdx = localOffsets[bucket]++;
        dstKeys[destIdx] = srcKeys[i];
        dstIndices[destIdx] = srcIndices[i];
    }
}

void lockFreeRadixSort(uint32_t*& keys, uint32_t*& indices, size_t n, unsigned int numThreads) {
    uint32_t* tmpKeys = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    uint32_t* tmpIndices = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    
    uint32_t *srcK = keys, *dstK = tmpKeys;
    uint32_t *srcI = indices, *dstI = tmpIndices;
    
    std::vector<RadixHist> hists(numThreads);
    std::vector<std::thread> threads;
    size_t blockSize = (n + numThreads - 1) / numThreads;
    
    // Pre-allocate per-thread offset arrays
    std::vector<std::vector<uint32_t>> threadOffsets(numThreads, std::vector<uint32_t>(256));
    
    for (int shift = 0; shift < 32; shift += 8) {
        // Phase 1: Parallel histogram
        threads.clear();
        for (unsigned int t = 0; t < numThreads; t++) {
            size_t start = t * blockSize;
            size_t end = std::min(start + blockSize, n);
            if (start < n) threads.emplace_back(radixCountChunk, srcK, start, end, shift, &hists[t]);
        }
        for (auto& th : threads) th.join();
        
        // Phase 2: Compute exclusive prefix sums (serial, fast)
        uint32_t globalOffsets[256];
        uint32_t runningTotal = 0;
        for (int b = 0; b < 256; b++) {
            globalOffsets[b] = runningTotal;
            for (unsigned int t = 0; t < numThreads; t++) {
                runningTotal += hists[t].count[b];
            }
        }
        
        // Phase 3: Compute per-thread starting offsets (no atomics!)
        for (int b = 0; b < 256; b++) {
            uint32_t offset = globalOffsets[b];
            for (unsigned int t = 0; t < numThreads; t++) {
                threadOffsets[t][b] = offset;
                offset += hists[t].count[b];
            }
        }
        
        // Phase 4: Lock-free parallel scatter
        threads.clear();
        for (unsigned int t = 0; t < numThreads; t++) {
            size_t start = t * blockSize;
            size_t end = std::min(start + blockSize, n);
            if (start < n) {
                threads.emplace_back(radixScatterLockFree, 
                    srcK, srcI, dstK, dstI, start, end, shift, 
                    threadOffsets[t].data());
            }
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

void colorizeByTypeScalar(
    const uint32_t* RESTRICT indices,
    const uint16_t* RESTRICT srcTypes,
    float* RESTRICT dstColors,
    size_t start, size_t end
) {
    for (size_t i = start; i < end; i++) {
        uint32_t orgIdx = indices[i];
        uint16_t t = srcTypes[orgIdx];
        uint32_t cIdx = (t <= 7) ? t : 7;
        size_t out = i * 3;
        dstColors[out] = TYPE_COLORS_R[cIdx];
        dstColors[out + 1] = TYPE_COLORS_G[cIdx];
        dstColors[out + 2] = TYPE_COLORS_B[cIdx];
    }
}

// AVX2 SIMD COLORIZATION
SPATIAL_ASSEMBLER_AVX2_BMI2_TARGET void colorizeByTypeAVX2(
    const uint32_t* RESTRICT indices,
    const uint16_t* RESTRICT srcTypes,
    float* RESTRICT dstColors,
    size_t start, size_t end
) {
    size_t i = start;
    
    // Process 8 atoms at a time
    for (; i + 7 < end; i += 8) {
        // Gather type indices
        __m256i typeIdx = _mm256_set_epi32(
            std::min((uint32_t)srcTypes[indices[i+7]], 7u),
            std::min((uint32_t)srcTypes[indices[i+6]], 7u),
            std::min((uint32_t)srcTypes[indices[i+5]], 7u),
            std::min((uint32_t)srcTypes[indices[i+4]], 7u),
            std::min((uint32_t)srcTypes[indices[i+3]], 7u),
            std::min((uint32_t)srcTypes[indices[i+2]], 7u),
            std::min((uint32_t)srcTypes[indices[i+1]], 7u),
            std::min((uint32_t)srcTypes[indices[i+0]], 7u)
        );
        
        // Gather colors using indices
        __m256 r = _mm256_i32gather_ps(TYPE_COLORS_R, typeIdx, 4);
        __m256 g = _mm256_i32gather_ps(TYPE_COLORS_G, typeIdx, 4);
        __m256 b = _mm256_i32gather_ps(TYPE_COLORS_B, typeIdx, 4);
        
        // Interleave RGB (we need RGBRGBRGB... layout)
        // Store to temporary and interleave
        float rArr[8], gArr[8], bArr[8];
        _mm256_storeu_ps(rArr, r);
        _mm256_storeu_ps(gArr, g);
        _mm256_storeu_ps(bArr, b);
        
        for (int j = 0; j < 8; j++) {
            size_t out = (i + j) * 3;
            dstColors[out] = rArr[j];
            dstColors[out + 1] = gArr[j];
            dstColors[out + 2] = bArr[j];
        }
    }
    
    colorizeByTypeScalar(indices, srcTypes, dstColors, i, end);
}

void gatherPositions(
    const uint32_t* RESTRICT indices,
    const float* RESTRICT srcPos,
    float* RESTRICT dstPos,
    size_t start, size_t end
) {
    for (size_t i = start; i < end; i++) {
        uint32_t orgIdx = indices[i];
        size_t pSrc = orgIdx * 3;
        size_t pDst = i * 3;
        dstPos[pDst] = srcPos[pSrc];
        dstPos[pDst + 1] = srcPos[pSrc + 1];
        dstPos[pDst + 2] = srcPos[pSrc + 2];
    }
}

// DIRECT GLB BINARY GENERATION
// GLB Header structure
struct GLBHeader {
    uint32_t magic;      // 0x46546C67 = "glTF"
    uint32_t version;    // 2
    uint32_t length;     // Total file size
};

struct GLBChunk {
    uint32_t length;
    uint32_t type;       // 0x4E4F534A = JSON, 0x004E4942 = BIN
};

// Generate complete GLB buffer in C++
std::vector<uint8_t> generateGLBDirect(
    float* positions,
    float* colors,
    size_t atomCount,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    // Binary data sizes
    size_t posBytes = atomCount * 3 * sizeof(float);
    size_t colBytes = atomCount * 3 * sizeof(float);
    size_t binTotalBytes = posBytes + colBytes;
    
    // Pad binary to 4-byte alignment
    size_t binPadding = (4 - (binTotalBytes % 4)) % 4;
    size_t binChunkSize = binTotalBytes + binPadding;
    
    // Build JSON
    char json[2048];
    int jsonLen = snprintf(json, sizeof(json),
        R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Atoms"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"mode":0}],"name":"AtomCloud"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962}],"buffers":[{"byteLength":%zu}]})",
        atomCount, minX, minY, minZ, maxX, maxY, maxZ,
        atomCount,
        posBytes, posBytes, colBytes,
        binTotalBytes
    );
    
    // Pad JSON to 4-byte alignment
    size_t jsonPadding = (4 - (jsonLen % 4)) % 4;
    size_t jsonChunkSize = jsonLen + jsonPadding;
    
    // Total GLB size
    size_t totalSize = 12 + 8 + jsonChunkSize + 8 + binChunkSize;
    
    std::vector<uint8_t> glb(totalSize);
    uint8_t* ptr = glb.data();
    
    // GLB Header
    GLBHeader* header = (GLBHeader*)ptr;
    header->magic = 0x46546C67;
    header->version = 2;
    header->length = (uint32_t)totalSize;
    ptr += 12;
    
    // JSON Chunk
    GLBChunk* jsonChunk = (GLBChunk*)ptr;
    jsonChunk->length = (uint32_t)jsonChunkSize;
    jsonChunk->type = 0x4E4F534A; // JSON
    ptr += 8;
    memcpy(ptr, json, jsonLen);
    memset(ptr + jsonLen, ' ', jsonPadding); // Pad with spaces
    ptr += jsonChunkSize;
    
    // BIN Chunk
    GLBChunk* binChunk = (GLBChunk*)ptr;
    binChunk->length = (uint32_t)binChunkSize;
    binChunk->type = 0x004E4942; // BIN
    ptr += 8;
    memcpy(ptr, positions, posBytes);
    ptr += posBytes;
    memcpy(ptr, colors, colBytes);
    ptr += colBytes;
    memset(ptr, 0, binPadding); // Pad with zeros
    
    return glb;
}

// MAIN ENTRY POINT: GENERATE GLB FROM POSITIONS + TYPES
static napi_value GenerateGLB(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    
    // Get input arrays
    size_t posLen;
    float* srcPos;
    napi_get_typedarray_info(env, args[0], nullptr, &posLen, (void**)&srcPos, nullptr, nullptr);
    size_t n = posLen / 3;
    
    uint16_t* srcTypes;
    size_t typeLen;
    napi_get_typedarray_info(env, args[1], nullptr, &typeLen, (void**)&srcTypes, nullptr, nullptr);
    
    // Get bounds
    double minArr[3], maxArr[3];
    napi_value v;
    for (int i = 0; i < 3; i++) {
        napi_get_element(env, args[2], i, &v); napi_get_value_double(env, v, &minArr[i]);
        napi_get_element(env, args[3], i, &v); napi_get_value_double(env, v, &maxArr[i]);
    }
    
    const bool useFastCpuPath = hasSpatialAssemblerFastCpuFeatures();

    // Allocate output buffers
    float* outPos = (float*)aligned_malloc(n * 3 * sizeof(float));
    float* outCol = (float*)aligned_malloc(n * 3 * sizeof(float));
    uint32_t* keys = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    uint32_t* indices = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    
    unsigned int numThreads = getWorkerThreadCount(4);
    if (n < 100000) numThreads = 1;
    
    size_t blockSize = (n + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    
    // Phase 1: Compute Morton codes + initialize indices
    float invX = 1.0f / std::max(1e-10f, (float)(maxArr[0] - minArr[0]));
    float invY = 1.0f / std::max(1e-10f, (float)(maxArr[1] - minArr[1]));
    float invZ = 1.0f / std::max(1e-10f, (float)(maxArr[2] - minArr[2]));
    
    auto mortonWorker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            size_t p = i * 3;
            float x = (srcPos[p] - (float)minArr[0]) * invX;
            float y = (srcPos[p+1] - (float)minArr[1]) * invY;
            float z = (srcPos[p+2] - (float)minArr[2]) * invZ;
            
            uint32_t ux = std::min(1023u, std::max(0u, (uint32_t)(x * 1023.0f)));
            uint32_t uy = std::min(1023u, std::max(0u, (uint32_t)(y * 1023.0f)));
            uint32_t uz = std::min(1023u, std::max(0u, (uint32_t)(z * 1023.0f)));
            
            if (useFastCpuPath) {
                keys[i] = morton3D_BMI2(ux, uy, uz);
            } else {
                keys[i] = morton3DScalar(ux, uy, uz);
            }
            indices[i] = (uint32_t)i;
        }
    };
    
    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, n);
        if (start < n) threads.emplace_back(mortonWorker, start, end);
    }
    for (auto& th : threads) th.join();
    
    // Phase 2: Lock-free radix sort
    lockFreeRadixSort(keys, indices, n, numThreads);
    
    // Phase 3: Gather positions + colorize (parallel)
    threads.clear();
    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, n);
        if (start < n) {
            threads.emplace_back([&, start, end]() {
                gatherPositions(indices, srcPos, outPos, start, end);
                if (useFastCpuPath) {
                    colorizeByTypeAVX2(indices, srcTypes, outCol, start, end);
                } else {
                    colorizeByTypeScalar(indices, srcTypes, outCol, start, end);
                }
            });
        }
    }
    for (auto& th : threads) th.join();
    
    // Phase 4: Generate GLB binary
    std::vector<uint8_t> glb = generateGLBDirect(
        outPos, outCol, n,
        (float)minArr[0], (float)minArr[1], (float)minArr[2],
        (float)maxArr[0], (float)maxArr[1], (float)maxArr[2]
    );
    
    // Cleanup
    aligned_free(outPos);
    aligned_free(outCol);
    aligned_free(keys);
    aligned_free(indices);
    
    // Return as Buffer
    napi_value result;
    void* resultData;
    napi_create_buffer_copy(env, glb.size(), glb.data(), &resultData, &result);
    
    return result;
}

// STREAMING FILE-BASED GLB GENERATION (FOR 100M+ ATOMS)
// Write GLB directly to file using buffered I/O - bypasses Node.js 2GB buffer limit
bool writeGLBToFile(
    const char* outputPath,
    float* positions,
    float* colors,
    size_t atomCount,
    float minX, float minY, float minZ,
    float maxX, float maxY, float maxZ
) {
    FILE* f = fopen(outputPath, "wb");
    if (!f) return false;
    
    // Use 64MB write buffer for optimal disk I/O
    const size_t WRITE_BUFFER_SIZE = 64 * 1024 * 1024;
    char* writeBuffer = (char*)aligned_malloc(WRITE_BUFFER_SIZE);
    setvbuf(f, writeBuffer, _IOFBF, WRITE_BUFFER_SIZE);
    
    // Calculate sizes
    size_t posBytes = atomCount * 3 * sizeof(float);
    size_t colBytes = atomCount * 3 * sizeof(float);
    size_t binTotalBytes = posBytes + colBytes;
    size_t binPadding = (4 - (binTotalBytes % 4)) % 4;
    size_t binChunkSize = binTotalBytes + binPadding;
    
    // Build JSON header
    char json[2048];
    int jsonLen = snprintf(json, sizeof(json),
        R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Atoms"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"mode":0}],"name":"AtomCloud"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962}],"buffers":[{"byteLength":%zu}]})",
        atomCount, minX, minY, minZ, maxX, maxY, maxZ,
        atomCount,
        posBytes, posBytes, colBytes,
        binTotalBytes
    );
    
    size_t jsonPadding = (4 - (jsonLen % 4)) % 4;
    size_t jsonChunkSize = jsonLen + jsonPadding;
    uint32_t totalSize = 12 + 8 + jsonChunkSize + 8 + binChunkSize;
    
    // Write GLB Header (12 bytes)
    uint32_t header[3] = { 0x46546C67, 2, totalSize };
    fwrite(header, 1, 12, f);
    
    // Write JSON Chunk header (8 bytes)
    uint32_t jsonChunkHeader[2] = { (uint32_t)jsonChunkSize, 0x4E4F534A };
    fwrite(jsonChunkHeader, 1, 8, f);
    
    // Write JSON content + padding
    fwrite(json, 1, jsonLen, f);
    if (jsonPadding > 0) {
        char pad[4] = {' ', ' ', ' ', ' '};
        fwrite(pad, 1, jsonPadding, f);
    }
    
    // Write BIN Chunk header (8 bytes)
    uint32_t binChunkHeader[2] = { (uint32_t)binChunkSize, 0x004E4942 };
    fwrite(binChunkHeader, 1, 8, f);
    
    // Stream positions directly to file (no buffer limit)
    fwrite(positions, 1, posBytes, f);
    
    // Stream colors directly to file
    fwrite(colors, 1, colBytes, f);
    
    // Binary padding
    if (binPadding > 0) {
        char pad[4] = {0, 0, 0, 0};
        fwrite(pad, 1, binPadding, f);
    }
    
    fclose(f);
    aligned_free(writeBuffer);
    return true;
}

static napi_value GenerateGLBToFile(napi_env env, napi_callback_info info) {
    size_t argc = 5;
    napi_value args[5];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    
    // Get input arrays
    size_t posLen;
    float* srcPos;
    napi_get_typedarray_info(env, args[0], nullptr, &posLen, (void**)&srcPos, nullptr, nullptr);
    size_t n = posLen / 3;
    
    uint16_t* srcTypes;
    size_t typeLen;
    napi_get_typedarray_info(env, args[1], nullptr, &typeLen, (void**)&srcTypes, nullptr, nullptr);
    
    // Get bounds
    double minArr[3], maxArr[3];
    napi_value v;
    for (int i = 0; i < 3; i++) {
        napi_get_element(env, args[2], i, &v); napi_get_value_double(env, v, &minArr[i]);
        napi_get_element(env, args[3], i, &v); napi_get_value_double(env, v, &maxArr[i]);
    }
    
    // Get output file path
    size_t pathLen;
    napi_get_value_string_utf8(env, args[4], nullptr, 0, &pathLen);
    std::string outputPath(pathLen + 1, '\0');
    napi_get_value_string_utf8(env, args[4], &outputPath[0], outputPath.size(), &pathLen);
    outputPath.resize(pathLen);
    
    const bool useFastCpuPath = hasSpatialAssemblerFastCpuFeatures();

    // Allocate work buffers
    float* outPos = (float*)aligned_malloc(n * 3 * sizeof(float));
    float* outCol = (float*)aligned_malloc(n * 3 * sizeof(float));
    uint32_t* keys = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    uint32_t* indices = (uint32_t*)aligned_malloc(n * sizeof(uint32_t));
    
    unsigned int numThreads = getWorkerThreadCount(4);
    if (n < 100000) numThreads = 1;
    
    size_t blockSize = (n + numThreads - 1) / numThreads;
    std::vector<std::thread> threads;
    
    // Phase 1: Morton codes + indices
    float invX = 1.0f / std::max(1e-10f, (float)(maxArr[0] - minArr[0]));
    float invY = 1.0f / std::max(1e-10f, (float)(maxArr[1] - minArr[1]));
    float invZ = 1.0f / std::max(1e-10f, (float)(maxArr[2] - minArr[2]));
    
    auto mortonWorker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            size_t p = i * 3;
            float x = (srcPos[p] - (float)minArr[0]) * invX;
            float y = (srcPos[p+1] - (float)minArr[1]) * invY;
            float z = (srcPos[p+2] - (float)minArr[2]) * invZ;
            
            uint32_t ux = std::min(1023u, std::max(0u, (uint32_t)(x * 1023.0f)));
            uint32_t uy = std::min(1023u, std::max(0u, (uint32_t)(y * 1023.0f)));
            uint32_t uz = std::min(1023u, std::max(0u, (uint32_t)(z * 1023.0f)));
            
            if (useFastCpuPath) {
                keys[i] = morton3D_BMI2(ux, uy, uz);
            } else {
                keys[i] = morton3DScalar(ux, uy, uz);
            }
            indices[i] = (uint32_t)i;
        }
    };
    
    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, n);
        if (start < n) threads.emplace_back(mortonWorker, start, end);
    }
    for (auto& th : threads) th.join();
    
    // Phase 2: Sort
    lockFreeRadixSort(keys, indices, n, numThreads);
    
    // Phase 3: Gather + colorize
    threads.clear();
    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, n);
        if (start < n) {
            threads.emplace_back([&, start, end]() {
                gatherPositions(indices, srcPos, outPos, start, end);
                if (useFastCpuPath) {
                    colorizeByTypeAVX2(indices, srcTypes, outCol, start, end);
                } else {
                    colorizeByTypeScalar(indices, srcTypes, outCol, start, end);
                }
            });
        }
    }
    for (auto& th : threads) th.join();
    
    // Phase 4: Write GLB directly to file (streaming, no buffer limit)
    bool success = writeGLBToFile(
        outputPath.c_str(),
        outPos, outCol, n,
        (float)minArr[0], (float)minArr[1], (float)minArr[2],
        (float)maxArr[0], (float)maxArr[1], (float)maxArr[2]
    );
    
    // Cleanup
    aligned_free(outPos);
    aligned_free(outCol);
    aligned_free(keys);
    aligned_free(indices);
    
    napi_value result;
    napi_get_boolean(env, success, &result);
    return result;
}

// APPLY PROPERTY COLORS (GRADIENT-BASED)
static napi_value ApplyPropertyColors(napi_env env, napi_callback_info info) {
    if (!GRADIENT_LUT_INIT) initGradientLUT();
    
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    size_t n;
    float* values;
    napi_get_typedarray_info(env, args[0], nullptr, &n, (void**)&values, nullptr, nullptr);

    double minVal, maxVal;
    int32_t type;
    napi_get_value_double(env, args[1], &minVal);
    napi_get_value_double(env, args[2], &maxVal);
    napi_get_value_int32(env, args[3], &type);
    if (type < 0 || type >= GRADIENT_TYPE_COUNT) type = 0;

    napi_value outBuf, outArr;
    void* outPtr;
    napi_create_arraybuffer(env, n * 3 * sizeof(float), &outPtr, &outBuf);
    napi_create_typedarray(env, napi_float32_array, n * 3, outBuf, 0, &outArr);
    float* colors = (float*)outPtr;

    float rangeInv = (maxVal == minVal) ? 0 : 1.0f / (float)(maxVal - minVal);
    float minF = (float)minVal;
    const float* lutBase = &GRADIENT_LUT[type * GRADIENT_LUT_SIZE * 3];

    unsigned int numThreads = getWorkerThreadCount();
    std::vector<std::thread> threads;
    size_t blockSize = (n + numThreads - 1) / numThreads;

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            float v = (values[i] - minF) * rangeInv;
            v = std::max(0.0f, std::min(1.0f, v));
            int idx = (int)(v * (GRADIENT_LUT_SIZE - 1)) * 3;
            size_t out = i * 3;
            colors[out] = lutBase[idx];
            colors[out + 1] = lutBase[idx + 1];
            colors[out + 2] = lutBase[idx + 2];
        }
    };

    for (unsigned int t = 0; t < numThreads; t++) {
        size_t start = t * blockSize;
        size_t end = std::min(start + blockSize, n);
        if (start < n) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();

    return outArr;
}

// GENERATE POINT CLOUD GLB (PRE-COLORED, VEC3 OR VEC4 COLORS)
static napi_value GeneratePointCloudGLB(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    
    size_t posLen, colLen;
    float* positions;
    float* colors;
    napi_get_typedarray_info(env, args[0], nullptr, &posLen, (void**)&positions, nullptr, nullptr);
    napi_get_typedarray_info(env, args[1], nullptr, &colLen, (void**)&colors, nullptr, nullptr);
    
    size_t atomCount = posLen / 3;
    bool isVec4 = (colLen == atomCount * 4);
    
    // Get bounds from args[2] and args[3]
    napi_value val;
    double minArr[3], maxArr[3];
    for (int i = 0; i < 3; i++) {
        napi_get_element(env, args[2], i, &val); napi_get_value_double(env, val, &minArr[i]);
        napi_get_element(env, args[3], i, &val); napi_get_value_double(env, val, &maxArr[i]);
    }
    
    size_t posBytes = posLen * sizeof(float);
    size_t colBytes = colLen * sizeof(float);
    size_t binTotal = posBytes + colBytes;
    size_t binPadding = (4 - (binTotal % 4)) % 4;
    
    const char* colorType = isVec4 ? "VEC4" : "VEC3";
    
    char json[2048];
    int jsonLen = snprintf(json, sizeof(json),
        R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Atoms"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"mode":0}],"name":"AtomCloud"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"%s"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962}],"buffers":[{"byteLength":%zu}]})",
        atomCount, minArr[0], minArr[1], minArr[2], maxArr[0], maxArr[1], maxArr[2],
        atomCount, colorType,
        posBytes, posBytes, colBytes,
        binTotal
    );
    
    size_t jsonPadding = (4 - (jsonLen % 4)) % 4;
    size_t totalSize = 12 + 8 + jsonLen + jsonPadding + 8 + binTotal + binPadding;
    
    std::vector<uint8_t> glb(totalSize);
    uint8_t* p = glb.data();
    
    // Header
    *(uint32_t*)p = 0x46546C67; p += 4;
    *(uint32_t*)p = 2; p += 4;
    *(uint32_t*)p = (uint32_t)totalSize; p += 4;
    
    // JSON chunk
    *(uint32_t*)p = jsonLen + jsonPadding; p += 4;
    *(uint32_t*)p = 0x4E4F534A; p += 4;
    memcpy(p, json, jsonLen); p += jsonLen;
    memset(p, 0x20, jsonPadding); p += jsonPadding;
    
    // BIN chunk
    *(uint32_t*)p = binTotal + binPadding; p += 4;
    *(uint32_t*)p = 0x004E4942; p += 4;
    memcpy(p, positions, posBytes); p += posBytes;
    memcpy(p, colors, colBytes); p += colBytes;
    memset(p, 0, binPadding);
    
    napi_value result;
    void* resultData;
    napi_create_buffer_copy(env, glb.size(), glb.data(), &resultData, &result);
    return result;
}

// TAUBIN SMOOTHING (IN-PLACE)
static napi_value TaubinSmooth(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    
    size_t posLen;
    float* positions;
    napi_get_typedarray_info(env, args[0], nullptr, &posLen, (void**)&positions, nullptr, nullptr);
    
    size_t idxLen;
    uint32_t* indices;
    napi_get_typedarray_info(env, args[1], nullptr, &idxLen, (void**)&indices, nullptr, nullptr);
    
    int32_t iterations;
    napi_get_value_int32(env, args[2], &iterations);
    
    if (iterations <= 0) {
        napi_value result;
        napi_get_boolean(env, true, &result);
        return result;
    }
    
    const float lambda = 0.5f;
    const float mu = -0.52f;
    const size_t vertexCount = posLen / 3;
    
    // Build adjacency list
    std::vector<std::vector<uint32_t>> adj(vertexCount);
    for (size_t i = 0; i < idxLen; i += 3) {
        uint32_t v0 = indices[i], v1 = indices[i+1], v2 = indices[i+2];
        adj[v0].push_back(v1); adj[v0].push_back(v2);
        adj[v1].push_back(v0); adj[v1].push_back(v2);
        adj[v2].push_back(v0); adj[v2].push_back(v1);
    }
    
    // Remove duplicates
    for (auto& neighbors : adj) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    
    float* temp = (float*)aligned_malloc(posLen * sizeof(float));
    
    for (int iter = 0; iter < iterations; iter++) {
        // Lambda pass
        for (size_t i = 0; i < vertexCount; i++) {
            size_t i3 = i * 3;
            auto& nbrs = adj[i];
            if (nbrs.empty()) {
                temp[i3] = positions[i3];
                temp[i3+1] = positions[i3+1];
                temp[i3+2] = positions[i3+2];
                continue;
            }
            float avgX = 0, avgY = 0, avgZ = 0;
            for (uint32_t n : nbrs) {
                size_t n3 = n * 3;
                avgX += positions[n3];
                avgY += positions[n3+1];
                avgZ += positions[n3+2];
            }
            float inv = 1.0f / nbrs.size();
            avgX *= inv; avgY *= inv; avgZ *= inv;
            temp[i3] = positions[i3] + lambda * (avgX - positions[i3]);
            temp[i3+1] = positions[i3+1] + lambda * (avgY - positions[i3+1]);
            temp[i3+2] = positions[i3+2] + lambda * (avgZ - positions[i3+2]);
        }
        
        // Mu pass
        for (size_t i = 0; i < vertexCount; i++) {
            size_t i3 = i * 3;
            auto& nbrs = adj[i];
            if (nbrs.empty()) {
                positions[i3] = temp[i3];
                positions[i3+1] = temp[i3+1];
                positions[i3+2] = temp[i3+2];
                continue;
            }
            float avgX = 0, avgY = 0, avgZ = 0;
            for (uint32_t n : nbrs) {
                size_t n3 = n * 3;
                avgX += temp[n3];
                avgY += temp[n3+1];
                avgZ += temp[n3+2];
            }
            float inv = 1.0f / nbrs.size();
            avgX *= inv; avgY *= inv; avgZ *= inv;
            positions[i3] = temp[i3] + mu * (avgX - temp[i3]);
            positions[i3+1] = temp[i3+1] + mu * (avgY - temp[i3+1]);
            positions[i3+2] = temp[i3+2] + mu * (avgZ - temp[i3+2]);
        }
    }
    
    aligned_free(temp);
    
    napi_value result;
    napi_get_boolean(env, true, &result);
    return result;
}

// GENERATE MESH GLB (WITH INDICES, NORMALS, COLORS, MATERIAL)
static napi_value GenerateMeshGLB(napi_env env, napi_callback_info info) {
    size_t argc = 7;
    napi_value args[7];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    
    size_t posLen, normLen, idxLen, colLen = 0;
    float* positions;
    float* normals;
    void* indicesRaw;
    float* colors = nullptr;
    bool useU16 = false;
    
    napi_get_typedarray_info(env, args[0], nullptr, &posLen, (void**)&positions, nullptr, nullptr);
    napi_get_typedarray_info(env, args[1], nullptr, &normLen, (void**)&normals, nullptr, nullptr);
    
    napi_typedarray_type idxType;
    napi_get_typedarray_info(env, args[2], &idxType, &idxLen, &indicesRaw, nullptr, nullptr);
    useU16 = (idxType == napi_uint16_array);
    
    bool hasColors;
    napi_get_value_bool(env, args[3], &hasColors);
    if (hasColors) {
        napi_get_typedarray_info(env, args[4], nullptr, &colLen, (void**)&colors, nullptr, nullptr);
    }
    
    // Get bounds from args[5]
    napi_value boundsObj = args[5];
    napi_value val;
    double minX, minY, minZ, maxX, maxY, maxZ;
    napi_get_named_property(env, boundsObj, "minX", &val); napi_get_value_double(env, val, &minX);
    napi_get_named_property(env, boundsObj, "minY", &val); napi_get_value_double(env, val, &minY);
    napi_get_named_property(env, boundsObj, "minZ", &val); napi_get_value_double(env, val, &minZ);
    napi_get_named_property(env, boundsObj, "maxX", &val); napi_get_value_double(env, val, &maxX);
    napi_get_named_property(env, boundsObj, "maxY", &val); napi_get_value_double(env, val, &maxY);
    napi_get_named_property(env, boundsObj, "maxZ", &val); napi_get_value_double(env, val, &maxZ);
    
    // Get material from args[6]
    napi_value matObj = args[6];
    double baseR = 1, baseG = 1, baseB = 1, baseA = 1;
    double metallic = 0, roughness = 1;
    double emissiveR = 0, emissiveG = 0, emissiveB = 0;
    bool doubleSided = true;
    
    napi_value baseColorArr;
    if (napi_get_named_property(env, matObj, "baseColor", &baseColorArr) == napi_ok) {
        napi_get_element(env, baseColorArr, 0, &val); napi_get_value_double(env, val, &baseR);
        napi_get_element(env, baseColorArr, 1, &val); napi_get_value_double(env, val, &baseG);
        napi_get_element(env, baseColorArr, 2, &val); napi_get_value_double(env, val, &baseB);
        napi_get_element(env, baseColorArr, 3, &val); napi_get_value_double(env, val, &baseA);
    }
    napi_get_named_property(env, matObj, "metallic", &val); napi_get_value_double(env, val, &metallic);
    napi_get_named_property(env, matObj, "roughness", &val); napi_get_value_double(env, val, &roughness);
    napi_value emissiveArr;
    if (napi_get_named_property(env, matObj, "emissive", &emissiveArr) == napi_ok) {
        napi_get_element(env, emissiveArr, 0, &val); napi_get_value_double(env, val, &emissiveR);
        napi_get_element(env, emissiveArr, 1, &val); napi_get_value_double(env, val, &emissiveG);
        napi_get_element(env, emissiveArr, 2, &val); napi_get_value_double(env, val, &emissiveB);
    }
    napi_get_named_property(env, matObj, "doubleSided", &val); napi_get_value_bool(env, val, &doubleSided);
    
    size_t vertexCount = posLen / 3;
    size_t indexCount = idxLen;
    
    size_t posBytes = posLen * sizeof(float);
    size_t normBytes = normLen * sizeof(float);
    size_t colBytes = hasColors ? colLen * sizeof(float) : 0;
    size_t idxBytes = useU16 ? idxLen * sizeof(uint16_t) : idxLen * sizeof(uint32_t);
    
    size_t binTotal = posBytes + normBytes + colBytes + idxBytes;
    size_t binPadding = (4 - (binTotal % 4)) % 4;
    
    char json[8192];
    int jsonLen;
    
    const char* dsStr = doubleSided ? "true" : "false";
    
    if (hasColors) {
        jsonLen = snprintf(json, sizeof(json),
            R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Mesh"}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[%.4f,%.4f,%.4f,%.4f],"metallicFactor":%.4f,"roughnessFactor":%.4f},"emissiveFactor":[%.4f,%.4f,%.4f],"doubleSided":%s}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"COLOR_0":2},"indices":3,"material":0,"mode":4}],"name":"MeshGeometry"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"},{"bufferView":2,"componentType":5126,"count":%zu,"type":"VEC4"},{"bufferView":3,"componentType":%d,"count":%zu,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34963}],"buffers":[{"byteLength":%zu}]})",
            baseR, baseG, baseB, baseA, metallic, roughness, emissiveR, emissiveG, emissiveB, dsStr,
            vertexCount, minX, minY, minZ, maxX, maxY, maxZ,
            vertexCount, vertexCount,
            useU16 ? 5123 : 5125, indexCount,
            posBytes, posBytes, normBytes, posBytes + normBytes, colBytes, posBytes + normBytes + colBytes, idxBytes,
            binTotal
        );
    } else {
        jsonLen = snprintf(json, sizeof(json),
            R"({"asset":{"version":"2.0","generator":"Volt Native"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"Mesh"}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[%.4f,%.4f,%.4f,%.4f],"metallicFactor":%.4f,"roughnessFactor":%.4f},"emissiveFactor":[%.4f,%.4f,%.4f],"doubleSided":%s}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2,"material":0,"mode":4}],"name":"MeshGeometry"}],"accessors":[{"bufferView":0,"componentType":5126,"count":%zu,"type":"VEC3","min":[%.6f,%.6f,%.6f],"max":[%.6f,%.6f,%.6f]},{"bufferView":1,"componentType":5126,"count":%zu,"type":"VEC3"},{"bufferView":2,"componentType":%d,"count":%zu,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34962},{"buffer":0,"byteOffset":%zu,"byteLength":%zu,"target":34963}],"buffers":[{"byteLength":%zu}]})",
            baseR, baseG, baseB, baseA, metallic, roughness, emissiveR, emissiveG, emissiveB, dsStr,
            vertexCount, minX, minY, minZ, maxX, maxY, maxZ,
            vertexCount,
            useU16 ? 5123 : 5125, indexCount,
            posBytes, posBytes, normBytes, posBytes + normBytes, idxBytes,
            binTotal
        );
    }
    
    size_t jsonPadding = (4 - (jsonLen % 4)) % 4;
    size_t totalSize = 12 + 8 + jsonLen + jsonPadding + 8 + binTotal + binPadding;
    
    std::vector<uint8_t> glb(totalSize);
    uint8_t* p = glb.data();
    
    *(uint32_t*)p = 0x46546C67; p += 4;
    *(uint32_t*)p = 2; p += 4;
    *(uint32_t*)p = (uint32_t)totalSize; p += 4;
    
    *(uint32_t*)p = jsonLen + jsonPadding; p += 4;
    *(uint32_t*)p = 0x4E4F534A; p += 4;
    memcpy(p, json, jsonLen); p += jsonLen;
    memset(p, 0x20, jsonPadding); p += jsonPadding;
    
    *(uint32_t*)p = binTotal + binPadding; p += 4;
    *(uint32_t*)p = 0x004E4942; p += 4;
    memcpy(p, positions, posBytes); p += posBytes;
    memcpy(p, normals, normBytes); p += normBytes;
    if (hasColors) { memcpy(p, colors, colBytes); p += colBytes; }
    memcpy(p, indicesRaw, idxBytes); p += idxBytes;
    memset(p, 0, binPadding);
    
    napi_value result;
    void* resultData;
    napi_create_buffer_copy(env, glb.size(), glb.data(), &resultData, &result);
    return result;
}

// MODULE INIT
static napi_value Init(napi_env env, napi_value exports) {
    napi_value fn;
    
    napi_create_function(env, nullptr, 0, GenerateGLB, nullptr, &fn);
    napi_set_named_property(env, exports, "generateGLB", fn);
    
    napi_create_function(env, nullptr, 0, GenerateGLBToFile, nullptr, &fn);
    napi_set_named_property(env, exports, "generateGLBToFile", fn);
    
    napi_create_function(env, nullptr, 0, ApplyPropertyColors, nullptr, &fn);
    napi_set_named_property(env, exports, "applyPropertyColors", fn);
    
    napi_create_function(env, nullptr, 0, TaubinSmooth, nullptr, &fn);
    napi_set_named_property(env, exports, "taubinSmooth", fn);
    
    napi_create_function(env, nullptr, 0, GenerateMeshGLB, nullptr, &fn);
    napi_set_named_property(env, exports, "generateMeshGLB", fn);
    
    napi_create_function(env, nullptr, 0, GeneratePointCloudGLB, nullptr, &fn);
    napi_set_named_property(env, exports, "generatePointCloudGLB", fn);
    
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
