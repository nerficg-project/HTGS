#include "alpha_blend_global_ordering/fast_inference.h"
#include "alpha_blend_global_ordering/kernels/fast_inference.cuh"
#include "alpha_blend_global_ordering/buffer_utils.h"
#include "alpha_blend_global_ordering/config.h"
#include "shared_kernels.cuh"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>

void htgs::rasterization::alpha_blend_global_ordering::fast_inference(
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
    const int n_primitives,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const float near,
    const float far,
    const float scale_modifier,
    const bool to_chw)
{
    cudaMemcpyToSymbol(c_M3, M + 2, sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, cam_position, sizeof(float3), 0, cudaMemcpyDeviceToDevice);

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles) + 32;

    constexpr bool store_rgba = true, store_rgb_clamp_info = false;
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives, store_rgba, store_rgb_clamp_info));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgba, store_rgb_clamp_info);

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug_fast_inference) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    }
    else cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    kernels::fast_inference::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess),  config::block_size_preprocess>>>(
        positions,
        scales,
        rotations,
        opacities,
        sh_0,
        sh_rest,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.VPMT1,
        per_primitive_buffers.VPMT2,
        per_primitive_buffers.VPMT4,
        per_primitive_buffers.depth,
        per_primitive_buffers.rgba,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        total_sh_bases,
        near,
        far,
        scale_modifier
    );
    CHECK_CUDA(config::debug_fast_inference, "preprocess")

    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_primitives
    );
    CHECK_CUDA(config::debug_fast_inference, "cub::DeviceScan::InclusiveSum")

    int n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.offset + n_primitives - 1, sizeof(int), cudaMemcpyDeviceToHost);

    char* per_instance_buffers_blob = per_instance_buffers_func(required<PerInstanceBuffers>(n_instances, end_bit));
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, n_instances, end_bit);

    shared_kernels::create_instances_cu<<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.depth,
        per_instance_buffers.keys.Current(),
        per_instance_buffers.primitive_indices.Current(),
        grid.x,
        n_primitives
    );
    CHECK_CUDA(config::debug_fast_inference, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        per_instance_buffers.cub_workspace,
        per_instance_buffers.cub_workspace_size,
        per_instance_buffers.keys,
        per_instance_buffers.primitive_indices,
        n_instances,
        0, end_bit
    );
    CHECK_CUDA(config::debug_fast_inference, "cub::DeviceRadixSort::SortPairs")

    if constexpr (!config::debug_fast_inference) cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        shared_kernels::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances
        );
        CHECK_CUDA(config::debug_fast_inference, "extract_instance_ranges")
    }

    kernels::fast_inference::blend_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.VPMT1,
        per_primitive_buffers.VPMT2,
        per_primitive_buffers.VPMT4,
        per_primitive_buffers.rgba,
        image,
        width,
        height,
        grid.x,
        to_chw
    );
    CHECK_CUDA(config::debug_fast_inference, "blend")

}
