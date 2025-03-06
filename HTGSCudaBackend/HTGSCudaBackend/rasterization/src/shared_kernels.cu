#include "shared_kernels.cuh"
#include "helper_math.h"
#include <cstdint>

namespace htgs::rasterization::shared_kernels {

    template <typename KeyT>
    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        KeyT* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        const uint4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        for (uint y = screen_bounds.z; y < screen_bounds.w; ++y) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; ++x) {
                const KeyT tile_idx = y * grid_width + x;
                instance_keys[offset] = tile_idx;
                instance_primitive_indices[offset] = primitive_idx;
                offset++;
            }
        }
    }

    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        const float* primitive_depths,
        uint64_t* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        const uint primitive_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        const uint4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        const uint64_t depth_key = __float_as_uint(primitive_depths[primitive_idx]);
        for (uint y = screen_bounds.z; y < screen_bounds.w; ++y) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; ++x) {
                const uint64_t tile_idx = y * grid_width + x;
                instance_keys[offset] = (tile_idx << 32) | depth_key;
                instance_primitive_indices[offset] = primitive_idx;
                offset++;
            }
        }
    }

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }
    
    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const uint64_t instance_key = instance_keys[instance_idx];
        const uint instance_tile_idx = instance_key >> 32;
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint64_t previous_instance_key = instance_keys[instance_idx - 1];
            const uint previous_instance_tile_idx = previous_instance_key >> 32;
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    template __global__ void create_instances_cu<uint>(
        const uint*, const uint*, const uint4*, uint*, uint*, const uint, const uint);
    template __global__ void create_instances_cu<ushort>(
        const uint*, const uint*, const uint4*, ushort*, uint*, const uint, const uint);
    template __global__ void extract_instance_ranges_cu<uint>(
        const uint*, uint2*, const uint);
    template __global__ void extract_instance_ranges_cu<ushort>(
        const ushort*, uint2*, const uint);

}
