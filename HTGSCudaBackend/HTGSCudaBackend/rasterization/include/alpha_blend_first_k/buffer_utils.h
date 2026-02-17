#pragma once

#include "helper_math.h"
#include <cub/cub.cuh>

namespace htgs::rasterization::alpha_blend_first_k {

    template <typename T>
    static void obtain(char*& blob, T*& ptr, std::size_t count, std::size_t alignment) {
        std::size_t offset = reinterpret_cast<std::uintptr_t>(blob) + alignment - 1 & ~(alignment - 1);
        ptr = reinterpret_cast<T*>(offset);
        blob = reinterpret_cast<char*>(ptr + count);
    }

    template<typename T, typename... Args> 
	size_t required(size_t P, Args... args){
		char* size = nullptr;
		T::from_blob(size, P, args...);
		return ((size_t)size) + 128;
	}

    struct PerPrimitiveBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        uint* n_touched_tiles;
        uint* offset;
        uint4* screen_bounds;
        float4* VPMT1;
        float4* VPMT2;
        float4* VPMT4;
        float4* MT3;
        float3* rgb = nullptr;
        bool* rgb_clamp_info = nullptr;

        static PerPrimitiveBuffers from_blob(char*& blob, int n_primitives, bool store_rgb, bool store_rgb_clamp_info) {
            PerPrimitiveBuffers buffers;
            obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
            obtain(blob, buffers.offset, n_primitives, 128);
            obtain(blob, buffers.screen_bounds, n_primitives, 128);
            obtain(blob, buffers.VPMT1, n_primitives, 128);
            obtain(blob, buffers.VPMT2, n_primitives, 128);
            obtain(blob, buffers.VPMT4, n_primitives, 128);
            obtain(blob, buffers.MT3, n_primitives, 128);
            if (store_rgb) obtain(blob, buffers.rgb, n_primitives, 128);
            if (store_rgb_clamp_info) obtain(blob, buffers.rgb_clamp_info, n_primitives * 3, 128);
            cub::DeviceScan::InclusiveSum(
                nullptr, buffers.cub_workspace_size,
                buffers.n_touched_tiles, buffers.offset,
                n_primitives
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    template <typename KeyT>
    struct PerInstanceBuffers {
        size_t cub_workspace_size;
        char* cub_workspace;
        cub::DoubleBuffer<uint> primitive_indices;
        cub::DoubleBuffer<KeyT> keys;
    
        static PerInstanceBuffers from_blob(char*& blob, int n_instances, int end_bit) {
            PerInstanceBuffers buffers;
            KeyT* keys_current;
            obtain(blob, keys_current, n_instances, 128);
            KeyT* keys_alternate;
            obtain(blob, keys_alternate, n_instances, 128);
            buffers.keys = cub::DoubleBuffer<KeyT>(keys_current, keys_alternate);
            uint* primitive_indices_current;
            obtain(blob, primitive_indices_current, n_instances, 128);
            uint* primitive_indices_alternate;
            obtain(blob, primitive_indices_alternate, n_instances, 128);
            buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
            cub::DeviceRadixSort::SortPairs(
                nullptr, buffers.cub_workspace_size,
                buffers.keys, buffers.primitive_indices,
                n_instances,
                0, end_bit
            );
            obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
            return buffers;
        }
    };

    struct PerTileBuffers {
        uint2* instance_ranges;
    
        static PerTileBuffers from_blob(char*& blob, int n_tiles) {
            PerTileBuffers buffers;
            obtain(blob, buffers.instance_ranges, n_tiles, 128);
            return buffers;
        }
    };

    struct PerPixelBuffers {
        uint* primitive_indices_core;
        float4* grad_info_core;

        static PerPixelBuffers from_blob(char*& blob, int n_pixels, int K) {
            PerPixelBuffers buffers;
            obtain(blob, buffers.primitive_indices_core, n_pixels * K, 128);
            obtain(blob, buffers.grad_info_core, n_pixels * K, 128);
            return buffers;
        }
    };

}
