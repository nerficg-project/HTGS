#pragma once

#include "helper_math.h"
#include <functional>

namespace htgs::rasterization::alpha_blend_global_ordering {

    void update_max_weights(
        std::function<char* (size_t)> per_primitive_buffers_func,
        std::function<char* (size_t)> per_tile_buffers_func,
        std::function<char* (size_t)> per_instance_buffers_func,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float4* M,
        const float4* VPM,
        float* max_weights,
        const int n_primitives,
        const int width,
        const int height,
        const float near_plane,
        const float far_plane,
        const float weight_threshold);

}
