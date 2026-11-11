#pragma once
//
// smoothing.h - Taubin smoothing of triangle meshes (lambda/mu passes, shrink-free).
//
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace glbcore {

inline void taubin_smooth(float* positions, size_t vertexCount, const uint32_t* indices, size_t indexCount, int iterations) {
    if (iterations <= 0 || vertexCount == 0) return;
    const float lambda = 0.5f;
    const float mu = -0.52f;

    std::vector<std::vector<uint32_t>> neighbours(vertexCount);
    for (size_t i = 0; i + 2 < indexCount; i += 3) {
        uint32_t v0 = indices[i], v1 = indices[i + 1], v2 = indices[i + 2];
        if (v0 >= vertexCount || v1 >= vertexCount || v2 >= vertexCount) continue;
        neighbours[v0].push_back(v1); neighbours[v0].push_back(v2);
        neighbours[v1].push_back(v0); neighbours[v1].push_back(v2);
        neighbours[v2].push_back(v0); neighbours[v2].push_back(v1);
    }
    for (auto& list : neighbours) {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    }

    std::vector<float> temp(vertexCount * 3);
    auto pass = [&](const float* from, float* to, float factor) {
        for (size_t i = 0; i < vertexCount; i++) {
            const auto& list = neighbours[i];
            const float* p = &from[i * 3];
            float* out = &to[i * 3];
            if (list.empty()) {
                out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
                continue;
            }
            float ax = 0, ay = 0, az = 0;
            for (uint32_t n : list) {
                ax += from[n * 3];
                ay += from[n * 3 + 1];
                az += from[n * 3 + 2];
            }
            float inv = 1.0f / (float)list.size();
            out[0] = p[0] + factor * (ax * inv - p[0]);
            out[1] = p[1] + factor * (ay * inv - p[1]);
            out[2] = p[2] + factor * (az * inv - p[2]);
        }
    };
    for (int iteration = 0; iteration < iterations; iteration++) {
        pass(positions, temp.data(), lambda);
        pass(temp.data(), positions, mu);
    }
}

}  // namespace glbcore
