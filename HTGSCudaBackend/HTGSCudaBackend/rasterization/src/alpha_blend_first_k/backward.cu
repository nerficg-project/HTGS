#include "alpha_blend_first_k/backward.h"
#include "alpha_blend_first_k/kernels/backward.cuh"
#include "alpha_blend_first_k/buffer_utils.h"
#include "alpha_blend_first_k/config.h"
#include "rasterization_utils.h"
#include "utils.h"
#include "helper_math.h"
#include <variant>
#include <utility>

template <typename... Args>
void blend_k_templated(
    const int grid,
    const int block,
    const int K,
    Args&&... kernel_args)
{
    if (K >= 32) htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<32><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 16) htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<16><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 8) htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<8><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 4) htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<4><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else if (K >= 2) htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<2><<<grid, block>>>(std::forward<Args>(kernel_args)...);
    else htgs::rasterization::alpha_blend_first_k::kernels::backward::blend_cu<1><<<grid, block>>>(std::forward<Args>(kernel_args)...);
}

void htgs::rasterization::alpha_blend_first_k::backward(
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

    const int n_tiles = div_round_up(width, config::tile_width) * div_round_up(height, config::tile_height);
    const int end_bit = extract_end_bit(n_tiles);
    const int n_pixels = width * height;
    const int n_fragments = K * n_pixels;

    constexpr bool store_rgb = true, store_rgb_clamp_info = true;
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives, store_rgb, store_rgb_clamp_info);
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);
    PerPixelBuffers per_pixel_buffers = PerPixelBuffers::from_blob(per_pixel_buffers_blob, n_pixels, K);

    std::variant<PerInstanceBuffers<ushort>, PerInstanceBuffers<uint>> buffer_variant;
    if (end_bit <= 16) buffer_variant = PerInstanceBuffers<ushort>::from_blob(per_instance_buffers_blob, n_instances, end_bit);
    else buffer_variant = PerInstanceBuffers<uint>::from_blob(per_instance_buffers_blob, n_instances, end_bit);

    std::visit([&](auto& per_instance_buffers) {
        per_instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

        blend_k_templated(div_round_up(n_fragments, config::block_size_blend_backward), config::block_size_blend_backward, K,
            per_tile_buffers.instance_ranges,
            per_instance_buffers.primitive_indices.Current(),
            scales,
            rotations,
            opacities,
            per_primitive_buffers.VPMT1,
            per_primitive_buffers.VPMT2,
            per_primitive_buffers.VPMT4,
            per_pixel_buffers.primitive_indices_core,
            per_pixel_buffers.grad_info_core,
            grad_image,
            grad_positions,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_0,
            densification_info_helper,
            width,
            n_fragments,
            n_pixels
        );
        CHECK_CUDA(config::debug_backward, "blend_backward")

    }, buffer_variant);

    kernels::backward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess),  config::block_size_preprocess>>>(
        positions,
        sh_rest,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.rgb_clamp_info,
        densification_info_helper,
        grad_positions,
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
