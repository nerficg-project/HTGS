#pragma once

#include "helper_math.h"
#include <functional>

namespace htgs::rasterization::alpha_blend_global_ordering {

    void inference(
        std::function<char* (size_t)> per_primitive_buffers_func,
        std::function<char* (size_t)> per_tile_buffers_func,
        std::function<char* (size_t)> per_instance_buffers_func,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        const float4* M,
        const float4* VPM,
        const float3* cam_position,
        float* image,
        float* depth,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float near,
        const float far,
        const float scale_modifier,
        const bool to_chw,
        const bool use_median_depth);

}
