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

// Example 6: Dynamic memref with Tiling 
void example6_dynamic_memref_tiling();

// Example 7: tilling with hand made management of the workgroup
void example7_tiling_manual_workgroup();


} // namespace sandbox
