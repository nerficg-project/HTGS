#pragma once

#define DEF inline constexpr

namespace htgs::envmap::config {
    DEF bool debug = false;
    // block size constants
    DEF int block_size = 256;
}

namespace config = htgs::envmap::config;

#undef DEF