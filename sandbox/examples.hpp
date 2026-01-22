#pragma once

#include "matrix.hpp"
#include "common.hpp"

namespace sandbox {

// Example 1: Store block2d - demonstrates cooperative matrix store
void example1_store_block2d();

// Example 2: Add block2d - demonstrates cooperative matrix operations with apply
void example2_add_block2d();

// Example 3: Tiling with remainder checking
void example3_tiling_remainder();

// Example 4: Subview operations
void example4_subview();

// Example 5: Dynamic memref with dope vectors
void example5_dynamic_memref();

} // namespace sandbox
