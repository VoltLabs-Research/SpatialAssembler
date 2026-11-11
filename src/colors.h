#pragma once
//
// colors.h - the colours VOLT bakes into its models, as RGBA8.
//
// The only definition of the structure-type palette and of the property gradients: models carry
// their colours, so the viewer shows what the file holds and keeps no palette of its own.
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace glbcore {

// Atom type t (clamped to 7) takes row t.
static const uint8_t TYPE_COLORS[8][4] = {
    {128, 128, 128, 255},
    {255, 68, 68, 255},
    {68, 255, 68, 255},
    {68, 68, 255, 255},
    {255, 255, 68, 255},
    {255, 68, 255, 255},
    {68, 255, 255, 255},
    {153, 153, 153, 255}
};

inline void colors_by_type(const uint16_t* types, size_t count, uint8_t* rgba) {
    for (size_t i = 0; i < count; i++) {
        const uint8_t* color = TYPE_COLORS[std::min<uint16_t>(types[i], 7)];
        std::copy(color, color + 4, rgba + i * 4);
    }
}

// Gradient codes are a wire contract with the daemon (FilterEvaluator's GradientCode):
// 0 Viridis, 1 Plasma, 2 BlueRed, 3 Grayscale, 4 Magma, 5 Inferno, 6 Cividis, 7 RdBu,
// 8 Coolwarm, 9 Jet. Codes 0-3 predate this table and must keep their meaning. Stop colours
// mirror the client gradient previews (GradientPreview GRADIENT_CSS) so baked colours match
// the picker.
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

constexpr int GRADIENT_COUNT = 10;
constexpr int GRADIENT_LUT_SIZE = 1024;

static const GradientDef GRADIENTS[GRADIENT_COUNT] = {
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

inline uint8_t unit_to_u8(float value) {
    return (uint8_t)std::lround(std::max(0.0f, std::min(1.0f, value)) * 255.0f);
}

// RGBA8 lookup table of one gradient, sampled at GRADIENT_LUT_SIZE evenly spaced points.
inline std::vector<uint8_t> gradient_lut(int gradient) {
    const GradientDef& def = GRADIENTS[(gradient >= 0 && gradient < GRADIENT_COUNT) ? gradient : 0];
    std::vector<uint8_t> lut(GRADIENT_LUT_SIZE * 4);
    for (int i = 0; i < GRADIENT_LUT_SIZE; i++) {
        float t = i / (float)(GRADIENT_LUT_SIZE - 1);
        int hi = 1;
        while (hi < def.count - 1 && def.stops[hi].t < t) hi++;
        const GradientStop& s0 = def.stops[hi - 1];
        const GradientStop& s1 = def.stops[hi];
        float span = s1.t - s0.t;
        float local = span > 0.0f ? std::max(0.0f, std::min(1.0f, (t - s0.t) / span)) : 0.0f;
        for (int c = 0; c < 3; c++) {
            lut[i * 4 + c] = unit_to_u8(s0.rgb[c] + (s1.rgb[c] - s0.rgb[c]) * local);
        }
        lut[i * 4 + 3] = 255;
    }
    return lut;
}

// Colours count values by where they fall in [minValue, maxValue] on the gradient.
inline void property_colors(
    const float* values,
    size_t count,
    double minValue,
    double maxValue,
    int gradient,
    uint8_t* rgba
) {
    const std::vector<uint8_t> lut = gradient_lut(gradient);
    const float minF = (float)minValue;
    const float rangeInv = (maxValue == minValue) ? 0.0f : 1.0f / (float)(maxValue - minValue);

    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            float v = (values[i] - minF) * rangeInv;
            v = std::max(0.0f, std::min(1.0f, v));
            const uint8_t* color = &lut[(size_t)(v * (GRADIENT_LUT_SIZE - 1)) * 4];
            std::copy(color, color + 4, rgba + i * 4);
        }
    };

    unsigned int threads = count < 100000 ? 1 : std::max(1u, std::thread::hardware_concurrency());
    if (threads == 1) {
        worker(0, count);
        return;
    }
    std::vector<std::thread> pool;
    size_t block = (count + threads - 1) / threads;
    for (unsigned int t = 0; t < threads; t++) {
        size_t start = t * block;
        size_t end = std::min(count, start + block);
        if (start < end) pool.emplace_back(worker, start, end);
    }
    for (auto& thread : pool) thread.join();
}

}  // namespace glbcore
