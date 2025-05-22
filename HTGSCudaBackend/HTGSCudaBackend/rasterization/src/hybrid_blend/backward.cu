#include "hybrid_blend/backward.h"
#include "hybrid_blend/kernels/backward.cuh"
#include "hybrid_blend/buffer_utils.h"
#include "hybrid_blend/config.h"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <variant>
#include <utility>

template <typename... Args>
void blend_k_templated(
    const dim3& grid,
    const dim3& block,
    const int K,
    Args&&... kernel_args)
{
    if (K >= 32) htgs::rasterization::hybrid_blend::kernels::backward::blend_cu<32><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 16) htgs::rasterization::hybrid_blend::kernels::backward::blend_cu<16><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 8) htgs::rasterization::hybrid_blend::kernels::backward::blend_cu<8><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 4) htgs::rasterization::hybrid_blend::kernels::backward::blend_cu<4><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 2) htgs::rasterization::hybrid_blend::kernels::backward::blend_cu<2><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else htgs::rasterization::hybrid_blend::kernels::backward::blend_cu<1><<<grid, block>>>(std::forward<Args>(kernel_args)...);
}

void htgs::rasterization::hybrid_blend::backward(
    const float* grad_image,
    const float3* positions,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_rest,
    const float4* M,
    const float4* VPM,
    const float3* cam_position,
    char* per_primitive_buffers_blob,
    char* per_tile_buffers_blob,
    char* per_instance_buffers_blob,
    char* per_pixel_buffers_blob,
    float3* grad_positions,
    float3* grad_scales,
    float4* grad_rotations,
    float* grad_opacities,
    float3* grad_sh_0,
    float3* grad_sh_rest,
    float* grad_VPMT,
    float* densification_info,
    float* densification_info_helper,
    const int K,
    const int n_primitives,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const int n_instances,
    const int instance_primitive_indices_selector,
    const bool use_distance_scaling)
{
    if (use_distance_scaling) cudaMemcpyToSymbol(c_M3, M + 2, sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_VPM, VPM, 4 * sizeof(float4), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_cam_position, cam_position, sizeof(float3), 0, cudaMemcpyDeviceToDevice);

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles);
    const int n_pixels = width * height;

    constexpr bool store_rgba = true, store_rgb_clamp_info = true;
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgba, store_rgb_clamp_info);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerPixelBuffers per_pixel_buffers = PerPixelBuffers::from_blob(per_pixel_buffers_blob, n_pixels, K);

    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant;
    if (end_bit <= 16) buffer_variant = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    else buffer_variant = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob, n_instances, end_bit);

    std::visit([&](auto& per_instance_buffers) {
        per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

        blend_k_templated(grid, block, K,
            per_tile_buffers.instance_ranges,
            per_instance_buffers.primitive_indices.Current(),
            per_primitive_buffers.VPMT1,
            per_primitive_buffers.VPMT2,
            per_primitive_buffers.VPMT4,
            per_primitive_buffers.rgba,
            per_pixel_buffers.primitive_indices_core,
            per_pixel_buffers.grad_info_core,
            per_pixel_buffers.grad_info_tail,
            grad_image,
            grad_positions,
            grad_opacities,
            grad_sh_0,
            grad_VPMT,
            densification_info_helper,
            n_primitives,
            width,
            height,
            grid.x
        );
        CHECK_CUDA(config::debug_backward, "blend_backward")

    }, buffer_variant);

    kernels::backward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        positions,
        scales,
        rotations,
        sh_rest,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.rgb_clamp_info,
        grad_VPMT,
        densification_info_helper,
        grad_positions,
        grad_scales,
        grad_rotations,
        grad_sh_0,
        grad_sh_rest,
        densification_info,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        use_distance_scaling
    );
    CHECK_CUDA(config::debug_backward, "preprocess_backward")

}
