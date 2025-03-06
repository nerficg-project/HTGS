#pragma once

#include "helper_math.h"
#include "kernel_utils.cuh"
#include "alpha_blend_global_ordering/config.h"
#include <cooperative_groups.h>

namespace htgs::rasterization::alpha_blend_global_ordering::kernels::fast_inference {

    __global__ void preprocess_cu(
        const float3* positions,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_0,
        const float3* sh_rest,
        uint* primitive_n_touched_tiles,
        uint4* primitive_screen_bounds,
        float4* primitive_VPMT1,
        float4* primitive_VPMT2,
        float4* primitive_VPMT4,
        float* primitive_depths,
        float4* primitive_rgba,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float near,
        const float far,
        const float scale_modifier)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        // transform and cull
        const float3 position_world = positions[primitive_idx];
        const float opacity = opacities[primitive_idx];
        const float4 M3 = c_M3;
        uint n_touched_tiles;
        uint4 screen_bounds;
        float3 u, v, w;
        float4 VPMT1, VPMT2, VPMT4;
        float z;
        if (transform_and_cull(
            scales, rotations,
            position_world, opacity, M3,
            n_touched_tiles, screen_bounds, u, v, w, VPMT1, VPMT2, VPMT4, z,
            primitive_idx, grid_width, grid_height, config::tile_width, config::tile_height,
            near, far, config::min_alpha_threshold_rcp, scale_modifier
        )) return;

        // write intermediate results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = screen_bounds;
        primitive_VPMT1[primitive_idx] = VPMT1;
        primitive_VPMT2[primitive_idx] = VPMT2;
        primitive_VPMT4[primitive_idx] = VPMT4;
        primitive_depths[primitive_idx] = z;

        // compute view-dependent color
        const float3 rgb = convert_sh_to_rgb<false>(
            sh_0,
            sh_rest,
            nullptr,
            position_world,
            n_primitives,
            primitive_idx,
            active_sh_bases,
            total_sh_bases
        );
        primitive_rgba[primitive_idx] = make_float4(rgb, opacity);

    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float4* primitive_VPMT1,
        const float4* primitive_VPMT2,
        const float4* primitive_VPMT4,
        const float4* primitive_rgba,
        float* image,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool output_chw)
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
        __shared__ float4 collected_VPMT1[config::block_size_blend], collected_VPMT2[config::block_size_blend], collected_VPMT4[config::block_size_blend];
        __shared__ float3 collected_rgb[config::block_size_blend];
        __shared__ float collected_opacity[config::block_size_blend];
        // initialize local storage
        float3 rgb_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
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

                const float3 rgb = collected_rgb[j];
                rgb_pixel += blending_weight * rgb;

                transmittance *= 1.0f - alpha;
                if (transmittance < config::transmittance_threshold) done = true;
            }
        }
        if (inside) {
            // store results
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            if (output_chw) {
                const int n_pixels = width * height;
                image[pixel_idx] = __saturatef(rgb_pixel.x);
                image[n_pixels + pixel_idx] = __saturatef(rgb_pixel.y);
                image[2 * n_pixels + pixel_idx] = __saturatef(rgb_pixel.z);
            }
            else {
                const int base_idx = 3 * pixel_idx;
                image[base_idx] = __saturatef(rgb_pixel.x);
                image[base_idx + 1] = __saturatef(rgb_pixel.y);
                image[base_idx + 2] = __saturatef(rgb_pixel.z);
            }
        }
    }

}
