#pragma once

#include "helper_math.h"

__device__ __constant__ float4 c_M3;
__device__ __constant__ float4 c_VPM[4];
__device__ __constant__ float3 c_cam_position;

struct Mat3x3 {
    float r11, r12, r13;
    float r21, r22, r23;
    float r31, r32, r33;
};

template<typename T>
__device__ void swap(
    T& a,
    T& b)
{
    T temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ Mat3x3 convert_quaterion_to_rotation_matrix(
    const float4& quaternion)
{
    auto [r, x, y, z] = quaternion;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float rx = r * x, ry = r * y, rz = r * z;
    return {
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - rz), 2.0f * (xz + ry),
        2.0f * (xy + rz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - rx),
        2.0f * (xz - ry), 2.0f * (yz + rx), 1.0f - 2.0f * (xx + yy)
    };
}

__forceinline__ __device__ float4 convert_quaterion_to_rotation_matrix_backward(
    const float4& quaternion,
    const Mat3x3& dL_dR)
{
    auto [r, x, y, z] = quaternion;
    const float dL_dR_r21_sub_r12 = dL_dR.r21 - dL_dR.r12;
    const float dL_dR_r21_add_r12 = dL_dR.r21 + dL_dR.r12;
    const float dL_dR_r13_sub_r31 = dL_dR.r13 - dL_dR.r31;
    const float dL_dR_r13_add_r31 = dL_dR.r13 + dL_dR.r31;
    const float dL_dR_r32_sub_r23 = dL_dR.r32 - dL_dR.r23;
    const float dL_dR_r32_add_r23 = dL_dR.r32 + dL_dR.r23;
    return {
        2.0f * (x * dL_dR_r32_sub_r23 + y * dL_dR_r13_sub_r31 + z * dL_dR_r21_sub_r12),
        2.0f * (r * dL_dR_r32_sub_r23 - 2.0f * x * (dL_dR.r22 + dL_dR.r33) + y * dL_dR_r21_add_r12 + z * dL_dR_r13_add_r31),
        2.0f * (r * dL_dR_r13_sub_r31 + x * dL_dR_r21_add_r12 - 2.0f * y * (dL_dR.r11 + dL_dR.r33) + z * dL_dR_r32_add_r23),
        2.0f * (r * dL_dR_r21_sub_r12 + x * dL_dR_r13_add_r31 + y * dL_dR_r32_add_r23 - 2.0f * z * (dL_dR.r11 + dL_dR.r22))
    };
}

__forceinline__ __device__ bool transform_and_cull(
    const float3* scales,
    const float4* rotations,
    const float3& position_world,
    const float& opacity,
    const float4& M3,
    uint& n_touched_tiles,
    uint4& screen_bounds,
    float3& u,
    float3& v,
    float3& w,
    float4& VPMT1,
    float4& VPMT2,
    float4& VPMT4,
    float& z,
    const uint primitive_idx,
    const uint grid_width,
    const uint grid_height,
    const uint tile_width,
    const uint tile_height,
    const float near,
    const float far,
    const float min_alpha_threshold_rcp,
    const float scale_modifier)
{
    // early near/far plane culling
    z = dot(make_float3(M3), position_world) + M3.w;
    if (z < near || z > far) return true;

    // load scale, rotation, and opacity
    const float3 scale = scales[primitive_idx];
    const float4 quaternion = rotations[primitive_idx];
    const Mat3x3 R = convert_quaterion_to_rotation_matrix(quaternion);

    // compute screen-space bounding box
    u = make_float3(R.r11 * scale.x, R.r21 * scale.x, R.r31 * scale.x) * scale_modifier;
    v = make_float3(R.r12 * scale.y, R.r22 * scale.y, R.r32 * scale.y) * scale_modifier;
    w = make_float3(R.r13 * scale.z, R.r23 * scale.z, R.r33 * scale.z) * scale_modifier;
    const float4 VPM4 = c_VPM[3];
    VPMT4 = make_float4(dot(make_float3(VPM4), u), dot(make_float3(VPM4), v), dot(make_float3(VPM4), w), dot(make_float3(VPM4), position_world) + VPM4.w);
    // tight cutoff for the used opacity threshold
    const float rho_cutoff = 2.0f * logf(opacity * min_alpha_threshold_rcp);
    const float4 d = make_float4(rho_cutoff, rho_cutoff, rho_cutoff, -1.0f);
    const float s = dot(d, VPMT4 * VPMT4);
    if (s == 0.0f) return true;
    const float4 f = (1.0f / s) * d;
    // start with z-extent in screen-space for exact near/far plane culling
    const float4 VPM3 = c_VPM[2];
    const float4 VPMT3 = make_float4(dot(make_float3(VPM3), u), dot(make_float3(VPM3), v), dot(make_float3(VPM3), w), dot(make_float3(VPM3), position_world) + VPM3.w);
    const float center_z = dot(f, VPMT3 * VPMT4);
    const float extent_z = sqrtf(fmaxf(center_z * center_z - dot(f, VPMT3 * VPMT3), 0.0f));
    const float z_min = center_z - extent_z;
    const float z_max = center_z + extent_z;
    if (z_min < -1.0f || z_max > 1.0f) return true;
    // now x/y-extent of the screen-space bounding box
    const float4 VPM1 = c_VPM[0];
    VPMT1 = make_float4(dot(make_float3(VPM1), u), dot(make_float3(VPM1), v), dot(make_float3(VPM1), w), dot(make_float3(VPM1), position_world) + VPM1.w);
    const float center_x = dot(f, VPMT1 * VPMT4);
    const float extent_x = sqrtf(fmaxf(center_x * center_x - dot(f, VPMT1 * VPMT1), 0.0f));
    const float4 VPM2 = c_VPM[1];
    VPMT2 = make_float4(dot(make_float3(VPM2), u), dot(make_float3(VPM2), v), dot(make_float3(VPM2), w), dot(make_float3(VPM2), position_world) + VPM2.w);
    const float center_y = dot(f, VPMT2 * VPMT4);
    const float extent_y = sqrtf(fmaxf(center_y * center_y - dot(f, VPMT2 * VPMT2), 0.0f));

    // compute screen-space bounding box in pixel coordinates (+0.5 to account for half-pixel shift in V)
    screen_bounds = make_uint4(
        min(grid_width, static_cast<uint>(max(0, __float2int_rd((center_x - extent_x + 0.5f) / tile_width)))), // x_min
        min(grid_width, static_cast<uint>(max(0, __float2int_ru((center_x + extent_x + 0.5f) / tile_width)))), // x_max
        min(grid_height, static_cast<uint>(max(0, __float2int_rd((center_y - extent_y + 0.5f) / tile_height)))), // y_min
        min(grid_height, static_cast<uint>(max(0, __float2int_ru((center_y + extent_y + 0.5f) / tile_height)))) // y_max
    );

    // compute number of potentially influenced tiles
    n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
    return n_touched_tiles == 0;
}

template <bool train_mode>
__forceinline__ __device__ float3 convert_sh_to_rgb(
    const float3* sh_0,
    const float3* sh_rest,
    [[maybe_unused]] bool* rgb_clamp_info,
    const float3& position_world,
    const uint n_primitives,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_sh_bases)
{
    // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
    float3 result = 0.5f + 0.28209479177387814f * sh_0[primitive_idx];
    if (active_sh_bases > 1) {
        const float3* coefficients_ptr = sh_rest + primitive_idx * total_sh_bases;
        auto [x, y, z] = normalize(position_world - c_cam_position);
        result = result + (-0.48860251190291987f * y) * coefficients_ptr[0]
                        + (0.48860251190291987f * z) * coefficients_ptr[1]
                        + (-0.48860251190291987f * x) * coefficients_ptr[2];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            result = result + (1.0925484305920792f * xy) * coefficients_ptr[3]
                            + (-1.0925484305920792f * yz) * coefficients_ptr[4]
                            + (0.94617469575755997f * zz - 0.31539156525251999f) * coefficients_ptr[5]
                            + (-1.0925484305920792f * xz) * coefficients_ptr[6]
                            + (0.54627421529603959f * xx - 0.54627421529603959f * yy) * coefficients_ptr[7];
            if (active_sh_bases > 9) {
                result = result + (0.59004358992664352f * y * (-3.0f * xx + yy)) * coefficients_ptr[8]
                                + (2.8906114426405538f * xy * z) * coefficients_ptr[9]
                                + (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * coefficients_ptr[10]
                                + (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * coefficients_ptr[11]
                                + (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * coefficients_ptr[12]
                                + (1.4453057213202769f * z * (xx - yy)) * coefficients_ptr[13]
                                + (0.59004358992664352f * x * (-xx + 3.0f * yy)) * coefficients_ptr[14];
            }
        }
    }
    if constexpr (train_mode) {
        rgb_clamp_info[primitive_idx] = result.x < 0;
        rgb_clamp_info[n_primitives + primitive_idx] = result.y < 0;
        rgb_clamp_info[2 * n_primitives + primitive_idx] = result.z < 0;
    }
    return {
        fmaxf(0.0f, result.x),
        fmaxf(0.0f, result.y),
        fmaxf(0.0f, result.z)
    };
}

__forceinline__ __device__ float3 convert_sh_to_rgb_backward(
    const float3* sh_rest,
    const bool* rgb_clamp_info,
    float3* grad_sh_0,
    float3* grads_sh_rest,
    const float3& position_world,
    const uint n_primitives,
    const uint primitive_idx,
    const uint active_sh_bases,
    const uint total_sh_bases)
{
    const int coefficients_base_idx = primitive_idx * total_sh_bases;
    const float3* coefficients_ptr = sh_rest + coefficients_base_idx;
    float3* grad_coefficients_ptr = grads_sh_rest + coefficients_base_idx;

    const float3 grad_rgb_raw = grad_sh_0[primitive_idx];
    const float3 grad_rgb = make_float3(
        rgb_clamp_info[primitive_idx] ? 0.0f : grad_rgb_raw.x,
        rgb_clamp_info[n_primitives + primitive_idx] ? 0.0f : grad_rgb_raw.y,
        rgb_clamp_info[2 * n_primitives + primitive_idx] ? 0.0f : grad_rgb_raw.z
    );

    grad_sh_0[primitive_idx] = 0.28209479177387814f * grad_rgb;
    float3 drgb_dposition = make_float3(0.0f);
    if (active_sh_bases > 1) {
        auto [x_raw, y_raw, z_raw] = position_world - c_cam_position;
        auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
        grad_coefficients_ptr[0] = (-0.48860251190291987f * y) * grad_rgb;
        grad_coefficients_ptr[1] = (0.48860251190291987f * z) * grad_rgb;
        grad_coefficients_ptr[2] = (-0.48860251190291987f * x) * grad_rgb;
        float3 grad_direction_x = -0.48860251190291987f * coefficients_ptr[2];
        float3 grad_direction_y = -0.48860251190291987f * coefficients_ptr[0];
        float3 grad_direction_z = 0.48860251190291987f * coefficients_ptr[1];
        if (active_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            grad_coefficients_ptr[3] = (1.0925484305920792f * xy) * grad_rgb;
            grad_coefficients_ptr[4] = (-1.0925484305920792f * yz) * grad_rgb;
            grad_coefficients_ptr[5] = (0.94617469575755997f * zz - 0.31539156525251999f) * grad_rgb;
            grad_coefficients_ptr[6] = (-1.0925484305920792f * xz) * grad_rgb;
            grad_coefficients_ptr[7] = (0.54627421529603959f * xx - 0.54627421529603959f * yy) * grad_rgb;
            grad_direction_x = grad_direction_x + (1.0925484305920792f * y) * coefficients_ptr[3]
                                                + (-1.0925484305920792f * z) * coefficients_ptr[6]
                                                + (1.0925484305920792f * x) * coefficients_ptr[7];
            grad_direction_y = grad_direction_y + (1.0925484305920792f * x) * coefficients_ptr[3]
                                                + (-1.0925484305920792f * z) * coefficients_ptr[4]
                                                + (-1.0925484305920792f * y) * coefficients_ptr[7];
            grad_direction_z = grad_direction_z + (-1.0925484305920792f * y) * coefficients_ptr[4]
                                                + (1.8923493915151202f * z) * coefficients_ptr[5]
                                                + (-1.0925484305920792f * x) * coefficients_ptr[6];
            if (active_sh_bases > 9) {
                grad_coefficients_ptr[8] = (0.59004358992664352f * y * (-3.0f * xx + yy)) * grad_rgb;
                grad_coefficients_ptr[9] = (2.8906114426405538f * xy * z) * grad_rgb;
                grad_coefficients_ptr[10] = (0.45704579946446572f * y * (1.0f - 5.0f * zz)) * grad_rgb;
                grad_coefficients_ptr[11] = (0.3731763325901154f * z * (5.0f * zz - 3.0f)) * grad_rgb;
                grad_coefficients_ptr[12] = (0.45704579946446572f * x * (1.0f - 5.0f * zz)) * grad_rgb;
                grad_coefficients_ptr[13] = (1.4453057213202769f * z * (xx - yy)) * grad_rgb;
                grad_coefficients_ptr[14] = (0.59004358992664352f * x * (-xx + 3.0f * yy)) * grad_rgb;
                grad_direction_x = grad_direction_x + (-3.5402615395598609f * xy) * coefficients_ptr[8]
                                                    + (2.8906114426405538f * yz) * coefficients_ptr[9]
                                                    + (0.45704579946446572f - 2.2852289973223288f * zz) * coefficients_ptr[12]
                                                    + (2.8906114426405538f * xz) * coefficients_ptr[13]
                                                    + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * coefficients_ptr[14];
                grad_direction_y = grad_direction_y + (-1.7701307697799304f * xx + 1.7701307697799304f * yy) * coefficients_ptr[8]
                                                    + (2.8906114426405538f * xz) * coefficients_ptr[9]
                                                    + (0.45704579946446572f - 2.2852289973223288f * zz) * coefficients_ptr[10]
                                                    + (-2.8906114426405538f * yz) * coefficients_ptr[13]
                                                    + (3.5402615395598609f * xy) * coefficients_ptr[14];
                grad_direction_z = grad_direction_z + (2.8906114426405538f * xy) * coefficients_ptr[9]
                                                    + (-4.5704579946446566f * yz) * coefficients_ptr[10]
                                                    + (5.597644988851731f * zz - 1.1195289977703462f) * coefficients_ptr[11]
                                                    + (-4.5704579946446566f * xz) * coefficients_ptr[12]
                                                    + (1.4453057213202769f * xx - 1.4453057213202769f * yy) * coefficients_ptr[13];
            }
        }

        const float3 grad_direction = make_float3(
            dot(grad_direction_x, grad_rgb),
            dot(grad_direction_y, grad_rgb),
            dot(grad_direction_z, grad_rgb)
        );
        const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
        const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
        const float norm_sq = xx_raw + yy_raw + zz_raw;
        drgb_dposition = make_float3(
            (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
            -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
            -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z
        ) * rsqrtf(norm_sq * norm_sq * norm_sq);
    }
    return drgb_dposition;
}
