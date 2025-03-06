#pragma once

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
        const uint n_primitives);

    __global__ void create_instances_cu(
        const uint* primitive_n_touched_tiles,
        const uint* primitive_offsets,
        const uint4* primitive_screen_bounds,
        const float* primitive_depths,
        uint64_t* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives);

    template <typename KeyT>
    __global__ void extract_instance_ranges_cu(
        const KeyT* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);
    
    __global__ void extract_instance_ranges_cu(
        const uint64_t* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances);

}
