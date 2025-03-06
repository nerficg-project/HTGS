#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace inpc::rasterization::oit_blend::config {
    // debugging constants
    DEF bool debug_forward = false;
    DEF bool debug_backward = false;
    DEF bool debug_inference = false;
    DEF bool debug_update_max_weights = false;
    DEF bool debug_fast_inference = false;
    // rendering constants
    DEF float transmittance_threshold = 1e-4f;
    DEF float max_fragment_alpha = 1.0f; // 3dgs uses 0.99f
    DEF float one_minus_alpha_eps = max_fragment_alpha == 1.0f ? 1e-8f : 0.0f;
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_cutoff_sq = 11.0825270903f; // logf(min_alpha_threshold_rcp * min_alpha_threshold_rcp)
    // block size constants
    DEF int block_size_preprocess = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int tile_width = 8;
    DEF int tile_height = 8;
    DEF int block_size_blend = tile_width * tile_height;
}

namespace config = inpc::rasterization::oit_blend::config;

#undef DEF
