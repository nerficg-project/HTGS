#pragma once

#include "helper_math.h"

namespace htgs::rasterization::alpha_blend_global_ordering {

    void backward(
        const float* grad_image,
        const float* image,
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_rest,
        const float4* M,
        const float4* VPM,
        const float3* cam_position,
        char* per_primitive_buffers_blob,
        char* per_tile_buffers_blob,
        char* per_instance_buffers_blob,
        float3* grad_positions,
        float3* grad_scales,
        float4* grad_rotations,
        float* grad_opacities,
        float3* grad_sh_0,
        float3* grad_sh_rest,
        float* densification_info,
        float* densification_info_helper,
        const int n_primitives,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const int n_instances,
        const int instance_primitive_indices_selector,
        const bool use_distance_scaling);

}
