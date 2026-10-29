#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
#include "alpha_blend_global_ordering/config.h"
#include <cooperative_groups.h>

namespace htgs::rasterization::alpha_blend_global_ordering::kernels::backward {

    template<bool aaa_mode>
    __global__ void preprocess_cu(
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_rest,
        const uint* primitive_n_touched_tiles,
        const bool* primitive_rgb_clamp_info,
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

        const float3 position_world = positions[primitive_idx];

        // anti-aliasing filter: the blend backward accumulated the gradients with respect to the
        // dilated scale and the dilated opacity, so the filter is chained back here
        float3 dfilter_dposition = make_float3(0.0f, 0.0f, 0.0f);
        if constexpr (aaa_mode) {
            const float3 original_scale = scales[primitive_idx];
            const float4 quaternion = rotations[primitive_idx];
            const Mat3x3 R = convert_quaternion_to_rotation_matrix(quaternion);
            const float4 M3 = c_M[2];
            const float z = dot(make_float3(M3), position_world) + M3.w;
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
            grad_opacities[primitive_idx] = dL_dopacity_filtered * dilation_factor;

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
            const float3 dL_dscale = grad_scales[primitive_idx];
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

            grad_scales[primitive_idx] = dL_dscale * original_scale / scale + dL_dfilter_helper * (det_mul_ray_var_dil * ddet_mul_ray_var_doriginal_scale - det_mul_ray_var * ddet_mul_ray_var_dil_doriginal_scale);

            // rotation/quaternion gradients from the dilated view direction
            const Mat3x3 dL_dR = {
                view_dir_world.x * dL_dview_dir_gauss.x, view_dir_world.x * dL_dview_dir_gauss.y, view_dir_world.x * dL_dview_dir_gauss.z,
                view_dir_world.y * dL_dview_dir_gauss.x, view_dir_world.y * dL_dview_dir_gauss.y, view_dir_world.y * dL_dview_dir_gauss.z,
                view_dir_world.z * dL_dview_dir_gauss.x, view_dir_world.z * dL_dview_dir_gauss.y, view_dir_world.z * dL_dview_dir_gauss.z
            };
            grad_rotations[primitive_idx] += convert_quaternion_to_rotation_matrix_backward(quaternion, dL_dR);
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
            float z_scale = 1.0f;
            if (use_distance_scaling) {
                const float4 M3 = c_M[2];
                z_scale = (dot(make_float3(M3), position_world) + M3.w) * 0.5f;
            }
            densification_info[primitive_idx] += 1.0f;
            densification_info[n_primitives + primitive_idx] += length(dL_dposition * z_scale);
            if (densification_info_helper != nullptr) densification_info[2 * n_primitives + primitive_idx] += z_scale * densification_info_helper[primitive_idx];
        }
        else {
            grad_positions[primitive_idx] += drgb_dposition + dfilter_dposition;
        }
    }

    template<bool aaa_mode>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float3* scales,
        const float4* rotations,
        const float4* primitive_VPMT1,
        const float4* primitive_VPMT2,
        const float4* primitive_VPMT4,
        const float4* primitive_rgba,
        const float* primitive_depths,
        const float* grad_image,
        const float* grad_alphas,
        const float* image,
        const float* alphas,
        float3* grad_positions,
        float3* grad_scales,
        float4* grad_rotations,
        float* grad_opacities,
        float3* grad_sh_0, // used to store dL_drgb
        float* densification_info,
        const uint width,
        const uint height,
        const uint grid_width,
        const float max_focal)
    {
        const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float pixel_x = __uint2float_rn(pixel_coords.x);
        const float pixel_y = __uint2float_rn(pixel_coords.y);
        // setup shared memory
        __shared__ uint collected_primitive_indices[config::block_size_blend];
        __shared__ float4 collected_VPMT1[config::block_size_blend], collected_VPMT2[config::block_size_blend], collected_VPMT4[config::block_size_blend];
        __shared__ float3 collected_rgb[config::block_size_blend];
        __shared__ float collected_opacity[config::block_size_blend];
        // initialize local storage
        float3 grad_rgb_pixel, rgb_pixel;
        float dL_dalpha_from_alpha_common;
        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            grad_rgb_pixel = make_float3(
                grad_image[pixel_idx],
                grad_image[n_pixels + pixel_idx],
                grad_image[2 * n_pixels + pixel_idx]
            );
            rgb_pixel = make_float3(
                image[pixel_idx],
                image[n_pixels + pixel_idx],
                image[2 * n_pixels + pixel_idx]
            );
            dL_dalpha_from_alpha_common = grad_alphas[pixel_idx] * (1.0f - alphas[pixel_idx]);
        }
        float3 rgb_pixel_new = make_float3(0.0f);
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_primitive_indices[thread_rank] = primitive_idx;
                collected_VPMT1[thread_rank] = primitive_VPMT1[primitive_idx];
                collected_VPMT2[thread_rank] = primitive_VPMT2[primitive_idx];
                collected_VPMT4[thread_rank] = primitive_VPMT4[primitive_idx];
                const float4 rgba = primitive_rgba[primitive_idx];
                collected_rgb[thread_rank] = make_float3(rgba.x, rgba.y, rgba.z);
                collected_opacity[thread_rank] = rgba.w;
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                const float4 VPMT1 = collected_VPMT1[j];
                const float4 VPMT2 = collected_VPMT2[j];
                const float4 VPMT4 = collected_VPMT4[j];
                const float4 plane_x_diag = VPMT1 - VPMT4 * pixel_x;
                const float4 plane_y_diag = VPMT2 - VPMT4 * pixel_y;
                const float3 plane_x_diag_normal = make_float3(plane_x_diag);
                const float3 plane_y_diag_normal = make_float3(plane_y_diag);
                const float3 m = plane_x_diag.w * plane_y_diag_normal - plane_x_diag_normal * plane_y_diag.w;
                const float3 d = cross(plane_x_diag_normal, plane_y_diag_normal);
                const float numerator_rho2 = dot(m, m);
                const float denominator = dot(d, d);
                if (numerator_rho2 > config::max_cutoff_sq * denominator) continue; // considering opacity requires log/sqrt -> slower
                const float denominator_rcp = 1.0f / denominator;
                const float G = expf(-0.5f * numerator_rho2 * denominator_rcp);
                const float opacity = collected_opacity[j];
                const float alpha = fminf(opacity * G, config::max_fragment_alpha);
                if (alpha < config::min_alpha_threshold) continue;
                const float blending_weight = transmittance * alpha;
                const uint primitive_idx = collected_primitive_indices[j];

                // color gradient
                const float3 dL_drgb = blending_weight * grad_rgb_pixel;
                if (dL_drgb.x != 0.0f) atomicAdd(&grad_sh_0[primitive_idx].x, dL_drgb.x);
                if (dL_drgb.y != 0.0f) atomicAdd(&grad_sh_0[primitive_idx].y, dL_drgb.y);
                if (dL_drgb.z != 0.0f) atomicAdd(&grad_sh_0[primitive_idx].z, dL_drgb.z);

                // opacity gradient
                const float3 rgb = collected_rgb[j];
                rgb_pixel_new += blending_weight * rgb;
                const float one_minus_alpha = 1.0f - alpha;
                const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
                const float dL_dalpha_from_rgb = dot(transmittance * rgb + (rgb_pixel_new - rgb_pixel) * one_minus_alpha_rcp, grad_rgb_pixel);
                const float dL_dalpha_from_alpha = dL_dalpha_from_alpha_common * one_minus_alpha_rcp;
                const float dL_dalpha = dL_dalpha_from_rgb + dL_dalpha_from_alpha;
                if (dL_dalpha != 0.0f) {
                    const float dL_dopacity = dL_dalpha * G;
                    atomicAdd(&grad_opacities[primitive_idx], dL_dopacity);

                    const float dL_dG = dL_dalpha * opacity;
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
                    const float3 dL_dVPMT_c1 = make_float3(dL_dVPMT1.x, dL_dVPMT2.x, dL_dVPMT4.x);
                    const float3 dL_dVPMT_c2 = make_float3(dL_dVPMT1.y, dL_dVPMT2.y, dL_dVPMT4.y);
                    const float3 dL_dVPMT_c3 = make_float3(dL_dVPMT1.z, dL_dVPMT2.z, dL_dVPMT4.z);
                    const float3 dL_dVPMT_c4 = make_float3(dL_dVPMT1.w, dL_dVPMT2.w, dL_dVPMT4.w);

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
                    atomicAdd(&grad_positions[primitive_idx].x, dL_dposition.x);
                    atomicAdd(&grad_positions[primitive_idx].y, dL_dposition.y);
                    atomicAdd(&grad_positions[primitive_idx].z, dL_dposition.z);

                    // load/re-compute scale and rotation
                    const float3 original_scale = scales[primitive_idx];
                    float3 scale = original_scale;
                    if constexpr (aaa_mode) {
                        // the dilated scale is rebuilt from the view-space depth the forward stored,
                        // so that the two passes cannot disagree on it
                        const float filter_scale_view = primitive_depths[primitive_idx] / max_focal;
                        const float3 variance = original_scale * original_scale + filter_scale_view * filter_scale_view * config::aa_kernel_size;
                        scale = make_float3(sqrtf(variance.x), sqrtf(variance.y), sqrtf(variance.z));
                    }
                    const float4 quaternion = rotations[primitive_idx];
                    const Mat3x3 R = convert_quaternion_to_rotation_matrix(quaternion);

                    // scale gradient
                    const float3 R_c1 = make_float3(R.r11, R.r21, R.r31);
                    const float3 R_c2 = make_float3(R.r12, R.r22, R.r32);
                    const float3 R_c3 = make_float3(R.r13, R.r23, R.r33);
                    const float3 dL_dscale = make_float3(
                        dot(make_float3(dot(make_float3(VPM1), R_c1), dot(make_float3(VPM2), R_c1), dot(make_float3(VPM4), R_c1)), dL_dVPMT_c1),
                        dot(make_float3(dot(make_float3(VPM1), R_c2), dot(make_float3(VPM2), R_c2), dot(make_float3(VPM4), R_c2)), dL_dVPMT_c2),
                        dot(make_float3(dot(make_float3(VPM1), R_c3), dot(make_float3(VPM2), R_c3), dot(make_float3(VPM4), R_c3)), dL_dVPMT_c3)
                    );
                    atomicAdd(&grad_scales[primitive_idx].x, dL_dscale.x);
                    atomicAdd(&grad_scales[primitive_idx].y, dL_dscale.y);
                    atomicAdd(&grad_scales[primitive_idx].z, dL_dscale.z);

                    // rotation gradient
                    const Mat3x3 dL_dR = {
                        dot(VPM_c1, dL_dVPMT_c1) * scale.x, dot(VPM_c1, dL_dVPMT_c2) * scale.y, dot(VPM_c1, dL_dVPMT_c3) * scale.z,
                        dot(VPM_c2, dL_dVPMT_c1) * scale.x, dot(VPM_c2, dL_dVPMT_c2) * scale.y, dot(VPM_c2, dL_dVPMT_c3) * scale.z,
                        dot(VPM_c3, dL_dVPMT_c1) * scale.x, dot(VPM_c3, dL_dVPMT_c2) * scale.y, dot(VPM_c3, dL_dVPMT_c3) * scale.z
                    };
                    const float4 dL_drotation = convert_quaternion_to_rotation_matrix_backward(quaternion, dL_dR);
                    atomicAdd(&grad_rotations[primitive_idx].x, dL_drotation.x);
                    atomicAdd(&grad_rotations[primitive_idx].y, dL_drotation.y);
                    atomicAdd(&grad_rotations[primitive_idx].z, dL_drotation.z);
                    atomicAdd(&grad_rotations[primitive_idx].w, dL_drotation.w);

                    // densification info
                    if (densification_info != nullptr) atomicAdd(&densification_info[primitive_idx], fabsf(dL_dposition.x) + fabsf(dL_dposition.y) + fabsf(dL_dposition.z));

                }

                transmittance *= one_minus_alpha;
                if (transmittance < config::transmittance_threshold) done = true;
            }
        }
    }

}
