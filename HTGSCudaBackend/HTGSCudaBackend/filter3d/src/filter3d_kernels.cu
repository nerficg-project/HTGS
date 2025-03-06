#include "filter3d_kernels.cuh"
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
        const float near,
        const float distance2filter)
    {
        const uint point_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (point_idx >= n_points) return;
        const float3 position_world = positions[point_idx];
        const float4 w2c_r3 = w2c[2];
        const float z = dot(make_float3(w2c_r3), position_world) + w2c_r3.w;
        if (z < near) return;
        const float4 w2c_r1 = w2c[0];
        const float x_clip = dot(make_float3(w2c_r1), position_world) + w2c_r1.w;
        if (x_clip < left * z || x_clip > right * z) return;
        const float4 w2c_r2 = w2c[1];
        const float y_clip = dot(make_float3(w2c_r2), position_world) + w2c_r2.w;
        if (y_clip < top * z || y_clip > bottom * z) return;
        const float filter_3d_new = distance2filter * z;
        if (filter_3d[point_idx] < filter_3d_new) return;
        filter_3d[point_idx] = filter_3d_new;
        visibility_mask[point_idx] = true;
    }

}
