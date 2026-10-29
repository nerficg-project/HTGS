#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
#include "hybrid_blend/config.h"
#include <cooperative_groups.h>

namespace htgs::rasterization::hybrid_blend::kernels::backward {

    template<bool aaa_mode>
    __global__ void preprocess_cu(
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_rest,
        const uint* primitive_n_touched_tiles,
        const bool* primitive_rgb_clamp_info,
        const float* grad_VPMT,
        const float* densification_info_helper,
        float3* grad_positions,
        float* grad_opacities,
        float3* grad_scales,
        float4* grad_rotations,
        float3* grad_sh_0,
        float3* grad_sh_rest,
        float* densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float max_focal,
        const bool use_distance_scaling)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        // third row of dL_dVPMT is all zeros and thus not stored anywhere
        // fourth column of dL_dVPMT was already accounted for in blend backward
        const float3 dL_dVPMT_c1 = make_float3(grad_VPMT[primitive_idx], grad_VPMT[3 * n_primitives + primitive_idx], grad_VPMT[6 * n_primitives + primitive_idx]);
        const float3 dL_dVPMT_c2 = make_float3(grad_VPMT[n_primitives + primitive_idx], grad_VPMT[4 * n_primitives + primitive_idx], grad_VPMT[7 * n_primitives + primitive_idx]);
        const float3 dL_dVPMT_c3 = make_float3(grad_VPMT[2 * n_primitives + primitive_idx], grad_VPMT[5 * n_primitives + primitive_idx], grad_VPMT[8 * n_primitives + primitive_idx]);

        // load/re-compute position, scale, and rotation
        const float3 position_world = positions[primitive_idx];
        const float3 original_scale = scales[primitive_idx];
        const float4 quaternion = rotations[primitive_idx];
        const Mat3x3 R = convert_quaternion_to_rotation_matrix(quaternion);

        // compute viewspace z of Gaussian mean
        const float4 M3 = c_M[2];
        const float z = dot(make_float3(M3), position_world) + M3.w;

        // third row and fourth column of VPM are not needed here
        const float3 VPM1 = make_float3(c_VPM[0]);
        const float3 VPM2 = make_float3(c_VPM[1]);
        const float3 VPM4 = make_float3(c_VPM[3]);
        const float3 VPM_c1 = make_float3(VPM1.x, VPM2.x, VPM4.x);
        const float3 VPM_c2 = make_float3(VPM1.y, VPM2.y, VPM4.y);
        const float3 VPM_c3 = make_float3(VPM1.z, VPM2.z, VPM4.z);

        // scale gradients from VPMT1[:3], VPMT2[:3], and VPMT4[:3]
        const float3 R_c1 = make_float3(R.r11, R.r21, R.r31);
        const float3 R_c2 = make_float3(R.r12, R.r22, R.r32);
        const float3 R_c3 = make_float3(R.r13, R.r23, R.r33);
        const float3 dL_dscale = make_float3(
            dot(make_float3(dot(VPM1, R_c1), dot(VPM2, R_c1), dot(VPM4, R_c1)), dL_dVPMT_c1),
            dot(make_float3(dot(VPM1, R_c2), dot(VPM2, R_c2), dot(VPM4, R_c2)), dL_dVPMT_c2),
            dot(make_float3(dot(VPM1, R_c3), dot(VPM2, R_c3), dot(VPM4, R_c3)), dL_dVPMT_c3)
        );

        // anti-aliasing filter
        float3 dfilter_dposition = make_float3(0.0f, 0.0f, 0.0f);
        if constexpr (aaa_mode) {
            const float filter_scale_view = z / max_focal; // can be negative but it's fine as only the squared value is used
            const float filter_variance_view = filter_scale_view * filter_scale_view * config::aa_kernel_size;
            const float3 original_variance = original_scale * original_scale;
            const float3 variance = original_variance + filter_variance_view;
            const float3 scale = make_float3(sqrtf(variance.x), sqrtf(variance.y), sqrtf(variance.z));
            const float3 view_dir_world_raw = position_world - c_cam_position;
            const float3 view_dir_world = normalize(view_dir_world_raw);
            const float3 view_dir_gauss = make_float3(
                R.r11 * view_dir_world.x + R.r21 * view_dir_world.y + R.r31 * view_dir_world.z,
                R.r12 * view_dir_world.x + R.r22 * view_dir_world.y + R.r32 * view_dir_world.z,
                R.r13 * view_dir_world.x + R.r23 * view_dir_world.y + R.r33 * view_dir_world.z
            );
            const float3 r = view_dir_gauss * view_dir_gauss;
            const float det_mul_ray_var = dot(r, make_float3(original_variance.y * original_variance.z, original_variance.z * original_variance.x, original_variance.x * original_variance.y));
            const float det_mul_ray_var_dil = dot(r, make_float3(variance.y * variance.z, variance.z * variance.x, variance.x * variance.y));
            const float dilation_factor = sqrtf(__saturatef(det_mul_ray_var / det_mul_ray_var_dil));
            const float dL_dopacity_filtered = grad_opacities[primitive_idx];
            const float dL_dopacity = dL_dopacity_filtered * dilation_factor;
            grad_opacities[primitive_idx] = dL_dopacity;

            const float dL_ddilation_factor = dL_dopacity_filtered * opacities[primitive_idx];
            const float dL_dfilter_helper = dL_ddilation_factor * dilation_factor / (det_mul_ray_var * det_mul_ray_var_dil); // TODO: det_mul_ray_var can be zero
            const float3 ddet_mul_ray_var_doriginal_scale = make_float3(
                original_scale.x * (r.y * original_variance.z + r.z * original_variance.y),
                original_scale.y * (r.x * original_variance.z + r.z * original_variance.x),
                original_scale.z * (r.x * original_variance.y + r.y * original_variance.x)
            );
            const float3 ddet_mul_ray_var_dil_dvariance = make_float3(
                r.y * variance.z + r.z * variance.y,
                r.x * variance.z + r.z * variance.x,
                r.x * variance.y + r.y * variance.x
            );
            const float3 ddet_mul_ray_var_dil_doriginal_scale = original_scale * ddet_mul_ray_var_dil_dvariance;
            const float dL_ddet_mul_ray_var_dil = dL_ddilation_factor * -dilation_factor / det_mul_ray_var_dil;
            // the dilated scale depends on the filter variance as well, with d scale_i / dfilter_variance_view = 1 / (2 * scale_i);
            // the factor of 2 cancels against the one omitted from d filter_variance_view / dz in dL_dz below
            const float dL_dfilter_variance_view = dL_ddet_mul_ray_var_dil * (ddet_mul_ray_var_dil_dvariance.x + ddet_mul_ray_var_dil_dvariance.y + ddet_mul_ray_var_dil_dvariance.z)
                + dL_dscale.x / scale.x + dL_dscale.y / scale.y + dL_dscale.z / scale.z;
            const float dL_dz = dL_dfilter_variance_view * filter_scale_view * config::aa_kernel_size / max_focal;
            const float3 ddet_mul_ray_var_dview_dir_gauss = make_float3(
                view_dir_gauss.x * original_variance.y * original_variance.z,
                view_dir_gauss.y * original_variance.x * original_variance.z,
                view_dir_gauss.z * original_variance.x * original_variance.y
            );
            const float3 ddet_mul_ray_var_dil_dview_dir_gauss = make_float3(
                view_dir_gauss.x * variance.y * variance.z,
                view_dir_gauss.y * variance.x * variance.z,
                view_dir_gauss.z * variance.x * variance.y
            );
            const float3 dL_dview_dir_gauss = dL_dfilter_helper * (det_mul_ray_var_dil * ddet_mul_ray_var_dview_dir_gauss - det_mul_ray_var * ddet_mul_ray_var_dil_dview_dir_gauss);

            const float3 dL_dview_dir_world = make_float3(
                R.r11 * dL_dview_dir_gauss.x + R.r12 * dL_dview_dir_gauss.y + R.r13 * dL_dview_dir_gauss.z,
                R.r21 * dL_dview_dir_gauss.x + R.r22 * dL_dview_dir_gauss.y + R.r23 * dL_dview_dir_gauss.z,
                R.r31 * dL_dview_dir_gauss.x + R.r32 * dL_dview_dir_gauss.y + R.r33 * dL_dview_dir_gauss.z
            );

            const float x_raw = view_dir_world_raw.x, y_raw = view_dir_world_raw.y, z_raw = view_dir_world_raw.z;
            const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
            const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
            const float norm_sq = xx_raw + yy_raw + zz_raw;

            dfilter_dposition = dL_dz * make_float3(M3) + make_float3(
                (yy_raw + zz_raw) * dL_dview_dir_world.x - xy_raw * dL_dview_dir_world.y - xz_raw * dL_dview_dir_world.z,
                -xy_raw * dL_dview_dir_world.x + (xx_raw + zz_raw) * dL_dview_dir_world.y - yz_raw * dL_dview_dir_world.z,
                -xz_raw * dL_dview_dir_world.x - yz_raw * dL_dview_dir_world.y + (xx_raw + yy_raw) * dL_dview_dir_world.z
            ) * rsqrtf(norm_sq * norm_sq * norm_sq);
        
            const float3 dL_doriginal_scale = dL_dscale * original_scale / scale + dL_dfilter_helper * (det_mul_ray_var_dil * ddet_mul_ray_var_doriginal_scale - det_mul_ray_var * ddet_mul_ray_var_dil_doriginal_scale);
            grad_scales[primitive_idx] = dL_doriginal_scale;

            // rotation/quaternion gradients from VPMT1[:3], VPMT2[:3], and VPMT4[:3]
            const Mat3x3 dL_dR = {
                dot(VPM_c1, dL_dVPMT_c1) * scale.x + view_dir_world.x * dL_dview_dir_gauss.x,
                dot(VPM_c1, dL_dVPMT_c2) * scale.y + view_dir_world.x * dL_dview_dir_gauss.y,
                dot(VPM_c1, dL_dVPMT_c3) * scale.z + view_dir_world.x * dL_dview_dir_gauss.z,
                dot(VPM_c2, dL_dVPMT_c1) * scale.x + view_dir_world.y * dL_dview_dir_gauss.x,
                dot(VPM_c2, dL_dVPMT_c2) * scale.y + view_dir_world.y * dL_dview_dir_gauss.y,
                dot(VPM_c2, dL_dVPMT_c3) * scale.z + view_dir_world.y * dL_dview_dir_gauss.z,
                dot(VPM_c3, dL_dVPMT_c1) * scale.x + view_dir_world.z * dL_dview_dir_gauss.x,
                dot(VPM_c3, dL_dVPMT_c2) * scale.y + view_dir_world.z * dL_dview_dir_gauss.y,
                dot(VPM_c3, dL_dVPMT_c3) * scale.z + view_dir_world.z * dL_dview_dir_gauss.z
            };
            const float4 dL_drotation = convert_quaternion_to_rotation_matrix_backward(quaternion, dL_dR);
            grad_rotations[primitive_idx] = dL_drotation;
        } else {
            grad_scales[primitive_idx] = dL_dscale;

            // rotation/quaternion gradients from VPMT1[:3], VPMT2[:3], and VPMT4[:3]
            const Mat3x3 dL_dR = {
                dot(VPM_c1, dL_dVPMT_c1) * original_scale.x, dot(VPM_c1, dL_dVPMT_c2) * original_scale.y, dot(VPM_c1, dL_dVPMT_c3) * original_scale.z,
                dot(VPM_c2, dL_dVPMT_c1) * original_scale.x, dot(VPM_c2, dL_dVPMT_c2) * original_scale.y, dot(VPM_c2, dL_dVPMT_c3) * original_scale.z,
                dot(VPM_c3, dL_dVPMT_c1) * original_scale.x, dot(VPM_c3, dL_dVPMT_c2) * original_scale.y, dot(VPM_c3, dL_dVPMT_c3) * original_scale.z
            };
            const float4 dL_drotation = convert_quaternion_to_rotation_matrix_backward(quaternion, dL_dR);
            grad_rotations[primitive_idx] = dL_drotation;
        }

        // sh/position gradients from view-dependent color
        const float3 drgb_dposition = convert_sh_to_rgb_backward(
            sh_rest,
            primitive_rgb_clamp_info,
            grad_sh_0,
            grad_sh_rest,
            position_world,
            n_primitives,
            primitive_idx,
            active_sh_bases,
            total_sh_bases
        );
        if (densification_info != nullptr) {
            const float3 dL_dposition = grad_positions[primitive_idx] + drgb_dposition + dfilter_dposition;
            grad_positions[primitive_idx] = dL_dposition;

            // create densification_info from position gradients
            const float z_scale = use_distance_scaling ? z * 0.5f : 1.0f;
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(dL_dposition * z_scale);
            if (densification_info_helper != nullptr) densification_info[2 * n_primitives + primitive_idx] += z_scale * densification_info_helper[primitive_idx];
        }
        else {
            grad_positions[primitive_idx] += drgb_dposition + dfilter_dposition;
        }
    }

    template <int K>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float4* primitive_VPMT1,
        const float4* primitive_VPMT2,
        const float4* primitive_VPMT4,
        const float4* primitive_rgba,
        const uint* pixel_primitive_indices_core,
        const float4* pixel_grad_info_core,
        const float* pixel_grad_info_tail,
        const float* grad_image,
        const float* grad_alphas,
        const float* alphas,
        float3* grad_positions,
        float* grad_opacities,
        float3* grad_sh_0,
        float* grad_VPMT,
        float* densification_info,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();

        // setup shared memory
        __shared__ uint collected_primitive_indices_core[K * config::block_size_blend];
        __shared__ float3 collected_grads_rgb_core[K * config::block_size_blend], collected_grad_color_tail_partial[config::block_size_blend], collected_grad_alpha_tail_c[config::block_size_blend];
        __shared__ float collected_grads_alpha_core[K * config::block_size_blend], collected_grad_alpha_tail_common[config::block_size_blend], collected_grad_alpha_tail_a[config::block_size_blend], collected_bg_grad_partial[config::block_size_blend];

        // load per-pixel gradient info into shared memory
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            const float3 grad_pixel = make_float3(
                grad_image[pixel_idx],
                grad_image[n_pixels + pixel_idx],
                grad_image[2 * n_pixels + pixel_idx]
            );
            // core
            const int base_idx = thread_rank * K;
            #pragma unroll
            for (int core_idx = 0; core_idx < K; ++core_idx) {
                const int current_idx = n_pixels * core_idx + pixel_idx;
                collected_primitive_indices_core[base_idx + core_idx] = pixel_primitive_indices_core[current_idx];
                const float4 precomputed_grad_core = pixel_grad_info_core[current_idx];
                collected_grads_rgb_core[base_idx + core_idx] = precomputed_grad_core.w * grad_pixel;
                collected_grads_alpha_core[base_idx + core_idx] = dot(make_float3(precomputed_grad_core), grad_pixel);
            }
            // tail
            const float3 rgb_tail = make_float3(
                pixel_grad_info_tail[pixel_idx],
                pixel_grad_info_tail[n_pixels + pixel_idx],
                pixel_grad_info_tail[2 * n_pixels + pixel_idx]
            );
            const float alpha_sum_tail_rcp = pixel_grad_info_tail[3 * n_pixels + pixel_idx];
            const float transmittance_tail = pixel_grad_info_tail[4 * n_pixels + pixel_idx];
            const float transmittance_core = pixel_grad_info_tail[5 * n_pixels + pixel_idx];
            const float weight_tail = transmittance_core * alpha_sum_tail_rcp;
            const float alpha_tail = 1.0f - transmittance_tail;
            const float full_weight_tail = weight_tail * alpha_tail;
            const float rgb_tail_dot_grad_pixel = dot(rgb_tail, grad_pixel);
            collected_grad_color_tail_partial[thread_rank] = full_weight_tail * grad_pixel;
            collected_grad_alpha_tail_common[thread_rank] = -full_weight_tail * alpha_sum_tail_rcp * rgb_tail_dot_grad_pixel;
            collected_grad_alpha_tail_c[thread_rank] = full_weight_tail * grad_pixel;
            collected_grad_alpha_tail_a[thread_rank] = weight_tail * transmittance_tail * rgb_tail_dot_grad_pixel;
            // grad from background color
            collected_bg_grad_partial[thread_rank] = grad_alphas[pixel_idx] * (1.0f - alphas[pixel_idx]);
        }

        // every thread accumulates gradients for one gaussian at a time
        uint current_primitive_idx;
        float dL_dopacity_accum, densification_info_accum, current_opacity;
        float3 dL_drgb_accum, dL_dposition_accum, dL_dVPMT1_accum, dL_dVPMT2_accum, dL_dVPMT4_accum, current_rgb;
        float4 current_VPMT1, current_VPMT2, current_VPMT4;

        // process Gaussians in tile
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            block.sync();
            if (current_fetch_idx < tile_range.y) {
                // fetch primitive info
                current_primitive_idx = instance_primitive_indices[current_fetch_idx];
                current_VPMT1 = primitive_VPMT1[current_primitive_idx];
                current_VPMT2 = primitive_VPMT2[current_primitive_idx];
                current_VPMT4 = primitive_VPMT4[current_primitive_idx];
                const float4 rgba = primitive_rgba[current_primitive_idx];
                current_rgb = make_float3(rgba.x, rgba.y, rgba.z);
                current_opacity = rgba.w;
                // initialize accumulators
                dL_dopacity_accum = 0.0f;
                dL_drgb_accum = make_float3(0.0f);
                dL_dposition_accum = make_float3(0.0f);
                dL_dVPMT1_accum = make_float3(0.0f);
                dL_dVPMT2_accum = make_float3(0.0f);
                dL_dVPMT4_accum = make_float3(0.0f);
                if (densification_info != nullptr) densification_info_accum = 0.0f;
            }
            block.sync();
            // iterate over pixels
            const uint2 start_pixel_coords = make_uint2(group_index.x * config::tile_width, group_index.y * config::tile_height);
            for (uint pixel_idx_in_tile = 0u; pixel_idx_in_tile < config::block_size_blend; ++pixel_idx_in_tile) {
                block.sync();
                const uint2 current_tile_coords = make_uint2(pixel_idx_in_tile % config::tile_width, pixel_idx_in_tile / config::tile_width);
                const uint2 current_pixel_coords = make_uint2(start_pixel_coords.x + current_tile_coords.x, start_pixel_coords.y + current_tile_coords.y);
                const bool current_inside = current_pixel_coords.x < width && current_pixel_coords.y < height;
                if (current_inside) {
                    const float pixel_x = __uint2float_rn(current_pixel_coords.x);
                    const float pixel_y = __uint2float_rn(current_pixel_coords.y);
                    const float4 plane_x_diag = current_VPMT1 - current_VPMT4 * pixel_x;
                    const float4 plane_y_diag = current_VPMT2 - current_VPMT4 * pixel_y;
                    const float3 plane_x_diag_normal = make_float3(plane_x_diag);
                    const float3 plane_y_diag_normal = make_float3(plane_y_diag);
                    const float3 m = plane_x_diag.w * plane_y_diag_normal - plane_x_diag_normal * plane_y_diag.w;
                    const float3 d = cross(plane_x_diag_normal, plane_y_diag_normal);
                    const float numerator_rho2 = dot(m, m);
                    const float denominator = dot(d, d);
                    if (numerator_rho2 > config::max_cutoff_sq * denominator) continue; // considering opacity requires log/sqrt -> slower
                    const float denominator_rcp = 1.0f / denominator;
                    const float G = expf(-0.5f * numerator_rho2 * denominator_rcp);
                    const float alpha = fminf(current_opacity * G, config::max_fragment_alpha);
                    if (alpha < config::min_alpha_threshold) continue;

                    // figure out if the fragment is in the pixel's core or tail
                    const int base_idx_shared = pixel_idx_in_tile * K;
                    int core_idx = -1;
                    #pragma unroll
                    for (int i = 0; i < K; ++i) {
                        if (current_primitive_idx == collected_primitive_indices_core[base_idx_shared + i]) core_idx = i;
                    }
                    const bool is_core = core_idx >= 0;
                    const int core_idx_shared = base_idx_shared + core_idx;

                    // color gradient
                    const float3 dL_drgb_core = is_core
                        ? collected_grads_rgb_core[core_idx_shared]
                        : make_float3(0.0f, 0.0f, 0.0f);
                    const float3 dL_drgb_tail = !is_core
                        ? collected_grad_color_tail_partial[pixel_idx_in_tile] * alpha
                        : make_float3(0.0f, 0.0f, 0.0f);
                    dL_drgb_accum += dL_drgb_core + dL_drgb_tail; // one of these will be zero

                    // opacity/position/VPMT gradients
                    const float one_minus_alpha_rcp = 1.0f / fmaxf(1.0f - alpha, config::one_minus_alpha_eps);
                    const float dL_dalpha_core = is_core
                        ? collected_grads_alpha_core[core_idx_shared]
                        : 0.0f;
                    const float dL_dalpha_tail = !is_core
                        ? collected_grad_alpha_tail_common[pixel_idx_in_tile] +
                          dot(current_rgb, collected_grad_alpha_tail_c[pixel_idx_in_tile]) +
                          collected_grad_alpha_tail_a[pixel_idx_in_tile] * one_minus_alpha_rcp
                        : 0.0f;
                    const float dL_dalpha_from_bg = collected_bg_grad_partial[pixel_idx_in_tile] * one_minus_alpha_rcp;
                    const float dL_dalpha = dL_dalpha_core + dL_dalpha_tail + dL_dalpha_from_bg; // one of these will be zero
                    if (dL_dalpha == 0.0f) continue;

                    const float dL_dopacity = dL_dalpha * G;
                    dL_dopacity_accum += dL_dopacity;

                    const float dL_dG = dL_dalpha * current_opacity;
                    const float dL_drho2 = dL_dG * -0.5f * G;
                    const float dL_dnum = dL_drho2 * denominator_rcp;
                    const float dL_ddenom = dL_drho2 * -numerator_rho2 * denominator_rcp * denominator_rcp;
                    const float3 dL_dm = dL_dnum * 2.0f * m;
                    const float3 dL_dd = dL_ddenom * 2.0f * d;

                    const float4 dL_dplane_x_diag = make_float4(
                        -plane_y_diag.w * dL_dm.x - plane_y_diag.z * dL_dd.y + plane_y_diag.y * dL_dd.z,
                        -plane_y_diag.w * dL_dm.y + plane_y_diag.z * dL_dd.x - plane_y_diag.x * dL_dd.z,
                        -plane_y_diag.w * dL_dm.z - plane_y_diag.y * dL_dd.x + plane_y_diag.x * dL_dd.y,
                        dot(plane_y_diag_normal, dL_dm)
                    );
                    const float4 dL_dplane_y_diag = make_float4(
                        plane_x_diag.w * dL_dm.x + plane_x_diag.z * dL_dd.y - plane_x_diag.y * dL_dd.z,
                        plane_x_diag.w * dL_dm.y - plane_x_diag.z * dL_dd.x + plane_x_diag.x * dL_dd.z,
                        plane_x_diag.w * dL_dm.z + plane_x_diag.y * dL_dd.x - plane_x_diag.x * dL_dd.y,
                        -dot(plane_x_diag_normal, dL_dm)
                    );
    
                    const float4 dL_dVPMT1 = dL_dplane_x_diag;
                    const float4 dL_dVPMT2 = dL_dplane_y_diag;
                    const float4 dL_dVPMT4 = -pixel_x * dL_dplane_x_diag - pixel_y * dL_dplane_y_diag;
                    const float3 dL_dVPMT_c4 = make_float3(dL_dVPMT1.w, dL_dVPMT2.w, dL_dVPMT4.w);
                    dL_dVPMT1_accum += make_float3(dL_dVPMT1);
                    dL_dVPMT2_accum += make_float3(dL_dVPMT2);
                    dL_dVPMT4_accum += make_float3(dL_dVPMT4);
    
                    // third row and fourth column of VPM are not needed here
                    const float4 VPM1 = c_VPM[0];
                    const float4 VPM2 = c_VPM[1];
                    const float4 VPM4 = c_VPM[3];
                    const float3 VPM_c1 = make_float3(VPM1.x, VPM2.x, VPM4.x);
                    const float3 VPM_c2 = make_float3(VPM1.y, VPM2.y, VPM4.y);
                    const float3 VPM_c3 = make_float3(VPM1.z, VPM2.z, VPM4.z);
    
                    // position gradient
                    const float3 dL_dposition = make_float3(
                        dot(VPM_c1, dL_dVPMT_c4),
                        dot(VPM_c2, dL_dVPMT_c4),
                        dot(VPM_c3, dL_dVPMT_c4)
                    );
                    dL_dposition_accum += dL_dposition;
                    if (densification_info != nullptr) densification_info_accum += fabsf(dL_dposition.x) + fabsf(dL_dposition.y) + fabsf(dL_dposition.z);
                }
            }
            block.sync();
            if (current_fetch_idx < tile_range.y) {
                // write gradients to global memory
                atomicAdd(&grad_sh_0[current_primitive_idx].x, dL_drgb_accum.x);
                atomicAdd(&grad_sh_0[current_primitive_idx].y, dL_drgb_accum.y);
                atomicAdd(&grad_sh_0[current_primitive_idx].z, dL_drgb_accum.z);
                atomicAdd(&grad_opacities[current_primitive_idx], dL_dopacity_accum);
                atomicAdd(&grad_positions[current_primitive_idx].x, dL_dposition_accum.x);
                atomicAdd(&grad_positions[current_primitive_idx].y, dL_dposition_accum.y);
                atomicAdd(&grad_positions[current_primitive_idx].z, dL_dposition_accum.z);
                atomicAdd(&grad_VPMT[current_primitive_idx], dL_dVPMT1_accum.x);
                atomicAdd(&grad_VPMT[n_primitives + current_primitive_idx], dL_dVPMT1_accum.y);
                atomicAdd(&grad_VPMT[2 * n_primitives + current_primitive_idx], dL_dVPMT1_accum.z);
                atomicAdd(&grad_VPMT[3 * n_primitives + current_primitive_idx], dL_dVPMT2_accum.x);
                atomicAdd(&grad_VPMT[4 * n_primitives + current_primitive_idx], dL_dVPMT2_accum.y);
                atomicAdd(&grad_VPMT[5 * n_primitives + current_primitive_idx], dL_dVPMT2_accum.z);
                atomicAdd(&grad_VPMT[6 * n_primitives + current_primitive_idx], dL_dVPMT4_accum.x);
                atomicAdd(&grad_VPMT[7 * n_primitives + current_primitive_idx], dL_dVPMT4_accum.y);
                atomicAdd(&grad_VPMT[8 * n_primitives + current_primitive_idx], dL_dVPMT4_accum.z);
                if (densification_info != nullptr) atomicAdd(&densification_info[current_primitive_idx], densification_info_accum);
            }
        }
    }

}
