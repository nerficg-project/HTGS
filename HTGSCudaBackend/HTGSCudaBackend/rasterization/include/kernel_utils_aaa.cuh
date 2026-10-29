#pragma once

#include "kernel_utils.cuh"

__device__ inline float max_contrib_plane(
    const float4 plane,
    const float4 VPMT1,
    const float4 VPMT2,
    const float4 VPMT3,
    const float4 VPMT4,
    float4& max_pos_screen)
{
    float norm = plane.w / (plane.x * plane.x + plane.y * plane.y + plane.z * plane.z);

    float4 max_pos_gauss = -norm * plane;
    max_pos_screen.x = dot(make_float3(VPMT1), make_float3(max_pos_gauss)) + VPMT1.w;
    max_pos_screen.y = dot(make_float3(VPMT2), make_float3(max_pos_gauss)) + VPMT2.w;
    max_pos_screen.z = dot(make_float3(VPMT3), make_float3(max_pos_gauss)) + VPMT3.w;
    max_pos_screen.w = dot(make_float3(VPMT4), make_float3(max_pos_gauss)) + VPMT4.w;
    return -max_pos_gauss.w;
}

__device__ inline float max_contrib_ray(
    const float4 plane_a,
    const float4 plane_b,
    float3& max_pos)
{
    float3 d = {
        plane_a.y * plane_b.z - plane_a.z * plane_b.y,
        plane_a.z * plane_b.x - plane_a.x * plane_b.z,
        plane_a.x * plane_b.y - plane_a.y * plane_b.x,
    };

    float3 m = {
        plane_a.w * plane_b.x - plane_a.x * plane_b.w,
        plane_a.w * plane_b.y - plane_a.y * plane_b.w,
        plane_a.w * plane_b.z - plane_a.z * plane_b.w,
    };

    float3 m_div_dd = m / dot(d, d);
    max_pos = cross(d, m_div_dd);
    return dot(m, m_div_dd);
}

__device__ inline float max_contrib_ray(
    const float4 plane_a,
    const float4 plane_b,
    const float4 VPMT1,
    const float4 VPMT2,
    const float4 VPMT3,
    const float4 VPMT4,
    float4& max_pos_screen)
{
    float3 max_pos_gauss;
    float contrib = max_contrib_ray(plane_a, plane_b, max_pos_gauss);
    max_pos_screen.x = dot(make_float3(VPMT1), max_pos_gauss) + VPMT1.w;
    max_pos_screen.y = dot(make_float3(VPMT2), max_pos_gauss) + VPMT2.w;
    max_pos_screen.z = dot(make_float3(VPMT3), max_pos_gauss) + VPMT3.w;
    max_pos_screen.w = dot(make_float3(VPMT4), max_pos_gauss) + VPMT4.w;
    return contrib;
}

__device__ inline bool in_screen_range(
    const float position,
    const float range_from,
    const float extent,
    const float position_h)
{
    const float d = position - range_from * position_h;
    return d > 0.0f && d < extent * position_h;
}

__device__ inline float max_contrib_gaussian_frustum(
     const float4 VPMT1,
     const float4 VPMT2,
     const float4 VPMT3,
     const float4 VPMT4,
     const float width,
     const float height)
{
    float max_contrib = __FLT_MAX__;

    const float3 range_from = make_float3(
        0.0f, // TODO optimize based on the fact that this is 0
        0.0f, // TODO optimize based on the fact that this is 0
        -1.0f
    );
    const float3 extent = make_float3(
        width - 1.0f,
        height - 1.0f,
        2.0f
    );

    const float3 d = make_float3(
        VPMT1.w - range_from.x * VPMT4.w,
        VPMT2.w - range_from.y * VPMT4.w,
        VPMT3.w - range_from.z * VPMT4.w
    );
    const bool between_x = d.x > 0.0f && d.x < extent.x * VPMT4.w;
    const bool between_y = d.y > 0.0f && d.y < extent.y * VPMT4.w;
    const bool between_z = d.z > 0.0f && d.z < extent.z * VPMT4.w;
    if (between_x && between_y && between_z) return 0.0f;

    const float dx = copysignf(extent.x * 0.5f, d.x - (extent.x * 0.5f * VPMT4.w));
    const float dy = copysignf(extent.y * 0.5f, d.y - (extent.y * 0.5f * VPMT4.w));

    const float4 closer_plane_x = VPMT1 - VPMT4 * (range_from.x + extent.x * 0.5f + dx);
    const float4 closer_plane_y = VPMT2 - VPMT4 * (range_from.y + extent.y * 0.5f + dy);

    float4 pos_screen;
    float contrib = max_contrib_plane(closer_plane_x, VPMT1, VPMT2, VPMT3, VPMT4, pos_screen);
    if (contrib < max_contrib && in_screen_range(pos_screen.y, range_from.y, extent.y, pos_screen.w) && in_screen_range(pos_screen.z, range_from.z, extent.z, pos_screen.w)) max_contrib = contrib;

    contrib = max_contrib_plane(closer_plane_y, VPMT1, VPMT2, VPMT3, VPMT4, pos_screen);
    if (contrib < max_contrib && in_screen_range(pos_screen.x, range_from.x, extent.x, pos_screen.w) && in_screen_range(pos_screen.z, range_from.z, extent.z, pos_screen.w)) max_contrib = contrib;

    contrib = max_contrib_ray(closer_plane_x, closer_plane_y, VPMT1, VPMT2, VPMT3, VPMT4, pos_screen);
    if (contrib < max_contrib && in_screen_range(pos_screen.z, range_from.z, extent.z, pos_screen.w)) max_contrib = contrib;

    const float4 other_plane_y = VPMT2 - VPMT4 * (range_from.y + extent.y * 0.5f - dy);
    contrib = max_contrib_ray(closer_plane_x, other_plane_y, VPMT1, VPMT2, VPMT3, VPMT4, pos_screen);
    if (contrib < max_contrib && in_screen_range(pos_screen.z, range_from.z, extent.z, pos_screen.w)) max_contrib = contrib;

    const float4 other_plane_x = VPMT1 - VPMT4 * (range_from.x + extent.x * 0.5f - dx);
    contrib = max_contrib_ray(other_plane_x, closer_plane_y, VPMT1, VPMT2, VPMT3, VPMT4, pos_screen);
    if (contrib < max_contrib && in_screen_range(pos_screen.z, range_from.z, extent.z, pos_screen.w)) max_contrib = contrib;

    return max_contrib;
 }

__device__ inline float normalize_angle(
    float theta)
{
    constexpr float pi = 3.141592654f;
    constexpr float two_pi = 2.0f * pi;
    theta = fmodf(theta, two_pi); // Wrap to (-2*pi, 2*pi)
    if (theta > pi) theta -= two_pi; // Adjust to (-pi, pi]
    if (theta <= -pi) theta += two_pi;
    return theta;
}

__device__ inline void compute_aabb_view(
    const float4 MT1,
    const float4 MT2,
    const float4 MT3,
    float focal_x, float focal_y,
    float center_x, float center_y,
    float cutoff,
    float2& center,
    float2& extent)
{
    constexpr float pi = 3.141592654f;
    constexpr float two_pi = 2.0f * pi;
    constexpr float max_theta = 0.5f * pi - 1e-5f;
    const float4 t = make_float4(cutoff, cutoff, cutoff, -1.0f);
    const float3 viewdir = normalize(make_float3(MT1.w, MT2.w, MT3.w));

    auto compute_theta = [&](int axis)
    {
        bool is_x = axis == 0;
        float theta_mu = atan2f((is_x ? viewdir.x : viewdir.y), viewdir.z);

        float4 MT_axis = is_x ? MT1 : MT2;
        float squared_axis = dot(t, MT_axis * MT_axis);
        float squared_z = dot(t, MT3 * MT3);
        float mid = dot(t, MT_axis * MT3);

        float inside_sqrt = mid * mid - squared_z * squared_axis;

        // set default to full screen
        float2 result = make_float2(-max_theta, max_theta);
        if (inside_sqrt > 0.0f)
        {
            float theta_mid = atan2f(-mid, -squared_z);
            float2 theta_axis_orig = make_float2(
                atan2f(-(mid + sqrtf(inside_sqrt)), -squared_z),
                atan2f(-(mid - sqrtf(inside_sqrt)), -squared_z)
            );
            float2 theta_axis = theta_axis_orig;

            // rotate to get correct order
            while (theta_axis.x > theta_mu)          theta_axis.x -= pi;
            while (theta_axis.x < (theta_mu - pi))   theta_axis.x += pi;

            while (theta_axis.y < theta_mu)          theta_axis.y += pi;
            while (theta_axis.y > (theta_mu + pi))   theta_axis.y -= pi;

            float2 theta_axis_norm = make_float2(normalize_angle(theta_axis.x), normalize_angle(theta_axis.y));
            if (theta_mu < 0.0f && fabsf(theta_axis_norm.x) < fabsf(theta_axis_norm.y)) theta_axis += two_pi;
            else if (theta_mu > 0.0f && fabsf(theta_axis_norm.y) < fabsf(theta_axis_norm.x)) theta_axis -= two_pi;

            // constrain angles to be in front of the camera
            result = make_float2(fmaxf(result.x, theta_axis.x), fminf(result.y, theta_axis.y));
        }

        return make_float2(tanf(result.x), tanf(result.y));
    };

    float2 bounds_x = center_x + focal_x * compute_theta(0);
    float2 bounds_y = center_y + focal_y * compute_theta(1);

    center = make_float2((bounds_x.y + bounds_x.x) / 2.0f, (bounds_y.y + bounds_y.x) / 2.0f);
    extent = make_float2((bounds_x.y - bounds_x.x) / 2.0f, (bounds_y.y - bounds_y.x) / 2.0f);
}

__device__ inline bool transform_and_cull_aaa(
    const float3* scales,
    const float4* rotations,
    const float3& position_world,
    uint& n_touched_tiles,
    uint4& screen_bounds,
    float3& u,
    float3& v,
    float3& w,
    float4& VPMT1,
    float4& VPMT2,
    float4& VPMT4,
    float4& MT3,
    float& opacity,
    const uint primitive_idx,
    const uint grid_width,
    const uint grid_height,
    const uint tile_width,
    const uint tile_height,
    const float width,
    const float height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float aa_kernel_size,
    const float min_alpha_threshold,
    const float min_alpha_threshold_rcp,
    const float scale_modifier)
{
    // load scale, rotation, and opacity
    const float3 original_scale = scales[primitive_idx];
    const float4 quaternion = rotations[primitive_idx];
    const Mat3x3 R = convert_quaternion_to_rotation_matrix(quaternion);

    // compute viewspace z of Gaussian mean
    const float4 M3 = c_M[2];
    const float z = dot(make_float3(M3), position_world) + M3.w;

    // anti-aliasing filter
    const float max_focal = fmaxf(focal_x, focal_y);
    const float filter_scale_view = z / max_focal; // can be negative but it's fine as only the squared value is used
    const float filter_variance_view = filter_scale_view * filter_scale_view * aa_kernel_size;
    const float3 original_variance = original_scale * original_scale;
    const float3 variance = original_variance + filter_variance_view;
    const float3 scale = make_float3(sqrtf(variance.x), sqrtf(variance.y), sqrtf(variance.z));
    const float3 view_dir_world = normalize(position_world - c_cam_position);
    const float3 view_dir_gauss = make_float3(
        R.r11 * view_dir_world.x + R.r21 * view_dir_world.y + R.r31 * view_dir_world.z,
        R.r12 * view_dir_world.x + R.r22 * view_dir_world.y + R.r32 * view_dir_world.z,
        R.r13 * view_dir_world.x + R.r23 * view_dir_world.y + R.r33 * view_dir_world.z
    );
    const float3 r = view_dir_gauss * view_dir_gauss;
    const float det_mul_ray_var = dot(r, make_float3(original_variance.y * original_variance.z, original_variance.z * original_variance.x, original_variance.x * original_variance.y));
    const float det_mul_ray_var_dil = dot(r, make_float3(variance.y * variance.z, variance.z * variance.x, variance.x * variance.y));
    const float dilation_factor = sqrtf(__saturatef(det_mul_ray_var / det_mul_ray_var_dil));
    opacity *= dilation_factor;
    if (opacity < min_alpha_threshold) return true;

    // tight cutoff for the used opacity threshold
    const float rho_cutoff = 2.0f * logf(opacity * min_alpha_threshold_rcp);

    // check if camera is inside the dilated Gaussian
    const float3 cam_position_shifted = c_cam_position - position_world;
    const float3 cam_position_gauss = make_float3(
        R.r11 * cam_position_shifted.x + R.r21 * cam_position_shifted.y + R.r31 * cam_position_shifted.z,
        R.r12 * cam_position_shifted.x + R.r22 * cam_position_shifted.y + R.r32 * cam_position_shifted.z,
        R.r13 * cam_position_shifted.x + R.r23 * cam_position_shifted.y + R.r33 * cam_position_shifted.z
    ) / scale;
    if (dot(cam_position_gauss, cam_position_gauss) < rho_cutoff) return true;

    // compute the transformation from normalized Gaussian to screen space
    u = make_float3(R.r11 * scale.x, R.r21 * scale.x, R.r31 * scale.x) * scale_modifier;
    v = make_float3(R.r12 * scale.y, R.r22 * scale.y, R.r32 * scale.y) * scale_modifier;
    w = make_float3(R.r13 * scale.z, R.r23 * scale.z, R.r33 * scale.z) * scale_modifier;
    const float4 VPM1 = c_VPM[0];
    const float4 VPM2 = c_VPM[1];
    const float4 VPM3 = c_VPM[2];
    const float4 VPM4 = c_VPM[3];
    VPMT1 = make_float4(dot(make_float3(VPM1), u), dot(make_float3(VPM1), v), dot(make_float3(VPM1), w), dot(make_float3(VPM1), position_world) + VPM1.w);
    VPMT2 = make_float4(dot(make_float3(VPM2), u), dot(make_float3(VPM2), v), dot(make_float3(VPM2), w), dot(make_float3(VPM2), position_world) + VPM2.w);
    const float4 VPMT3 = make_float4(dot(make_float3(VPM3), u), dot(make_float3(VPM3), v), dot(make_float3(VPM3), w), dot(make_float3(VPM3), position_world) + VPM3.w);
    VPMT4 = make_float4(dot(make_float3(VPM4), u), dot(make_float3(VPM4), v), dot(make_float3(VPM4), w), dot(make_float3(VPM4), position_world) + VPM4.w);

    // compute maximum contribution inside viewing frustum
    const float max_contribution = max_contrib_gaussian_frustum(VPMT1, VPMT2, VPMT3, VPMT4, width, height);
    if (max_contribution > rho_cutoff) return true;

    // compute bounding box center and extent
    const float4 M1 = c_M[0];
    const float4 M2 = c_M[1];
    const float4 MT1 = make_float4(dot(make_float3(M1), u), dot(make_float3(M1), v), dot(make_float3(M1), w), dot(make_float3(M1), position_world) + M1.w);
    const float4 MT2 = make_float4(dot(make_float3(M2), u), dot(make_float3(M2), v), dot(make_float3(M2), w), dot(make_float3(M2), position_world) + M2.w);
    MT3 = make_float4(dot(make_float3(M3), u), dot(make_float3(M3), v), dot(make_float3(M3), w), z);
    float2 center, extent;
    compute_aabb_view(MT1, MT2, MT3, focal_x, focal_y, center_x, center_y, rho_cutoff, center, extent);

    // compute screen-space bounding box in pixel coordinates
    screen_bounds = make_uint4(
        min(grid_width, static_cast<uint>(max(0, __float2int_rd((center.x - extent.x) / tile_width)))), // x_min
        min(grid_width, static_cast<uint>(max(0, __float2int_ru((center.x + extent.x) / tile_width)))), // x_max
        min(grid_height, static_cast<uint>(max(0, __float2int_rd((center.y - extent.y) / tile_height)))), // y_min
        min(grid_height, static_cast<uint>(max(0, __float2int_ru((center.y + extent.y) / tile_height)))) // y_max
    );

    // compute number of potentially influenced tiles
    n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
    return n_touched_tiles == 0;
}
