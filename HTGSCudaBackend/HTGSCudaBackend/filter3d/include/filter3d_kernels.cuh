#pragma once

#include "helper_math.h"

namespace htgs::filter3d {

    __global__ void update_3d_filter_cu(
        const float3* positions,
        const float4* w2c,
        float* filter_3d,
        bool* visibility_mask,
        const uint n_points,
        const float left,
        const float right,
        const float top,
        const float bottom,
        const float near_plane,
        const float distance2filter);

}
