#include "alpha_blend_global_ordering/update_max_weights.h"
#include "alpha_blend_global_ordering/kernels/update_max_weights.cuh"
#include "alpha_blend_global_ordering/buffer_utils.h"
#include "alpha_blend_global_ordering/config.h"
#include "shared_kernels.cuh"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>

void htgs::rasterization::alpha_blend_global_ordering::update_max_weights(
    std::function<char* (size_t)> per_primitive_buffers_func,
    std::function<char* (size_t)> per_tile_buffers_func,
    std::function<char* (size_t)> per_instance_buffers_func,
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float4* M,
    const float4* VPM,
    float* max_weights,
    const int n_primitives,
    const int width,
    const int height,
    const float near,
    const float far,
    const float weight_threshold)
{
    cudaMemcpyToSymbol(c_M3, M + 2, sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles) + 32;
    
    constexpr bool store_rgba = false, store_rgb_clamp_info = false;
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives, store_rgba, store_rgb_clamp_info));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgba, store_rgb_clamp_info);

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    static bool memset_stream_initialized = false;
    if (!memset_stream_initialized && !config::debug_update_max_weights) {
        cudaStreamCreate(&memset_stream);
        memset_stream_initialized = true;
    }
    cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);

    kernels::update_max_weights::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess),  config::block_size_preprocess>>>(
        positions,
        scales,
        rotations,
        opacities,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.VPMT1,
        per_primitive_buffers.VPMT2,
        per_primitive_buffers.VPMT4,
        per_primitive_buffers.depth,
        n_primitives,
        grid.x,
        grid.y,
        near,
        far
    );
    CHECK_CUDA(config::debug_update_max_weights, "preprocess")

    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_primitives
    );
    CHECK_CUDA(config::debug_update_max_weights, "cub::DeviceScan::InclusiveSum")

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
    CHECK_CUDA(config::debug_update_max_weights, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        per_instance_buffers.cub_workspace,
        per_instance_buffers.cub_workspace_size,
        per_instance_buffers.keys,
        per_instance_buffers.primitive_indices,
        n_instances,
        0, end_bit
    );
    CHECK_CUDA(config::debug_update_max_weights, "cub::DeviceRadixSort::SortPairs")

    if constexpr (!config::debug_update_max_weights) cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        shared_kernels::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances
        );
        CHECK_CUDA(config::debug_update_max_weights, "extract_instance_ranges")
    }

    kernels::update_max_weights::update_max_weights_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        opacities,
        per_primitive_buffers.VPMT1,
        per_primitive_buffers.VPMT2,
        per_primitive_buffers.VPMT4,
        reinterpret_cast<uint*>(max_weights),
        width,
        height,
        grid.x,
        weight_threshold
    );
    CHECK_CUDA(config::debug_update_max_weights, "update_max_weights")

}
