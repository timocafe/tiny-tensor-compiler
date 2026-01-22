#include "examples.hpp"

namespace sandbox {

extern const sycl::queue q;

void example1_store_block2d() {
    std::cout << "\n=== Example 1: Store Block2D ===\n";

    auto ctx = create_configured_context();
    execute_with_error_handling([&]() {
        // Initialize tensors
        matrix<float> A(32, 32);

        const std::string code = R"TinyTL(
func @store_block2d(%A: memref<f32x32x32> {alignment=128})
    attributes{subgroup_size=16,work_group_size=[16,1]} {
    parallel {
        %0 = constant 42.0 : coopmatrix<f32x8x8,matrix_acc>
        %1 = constant 23 : index
        %2 = constant 13 : index
        ; %1 and %2 specify the position to store the block
        cooperative_matrix_store %0, %A[%1,%2]
    }
})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "store_block2d");

        auto exe_range = tinytc::get_execution_range(kernel, sycl::range<3u>{1, 1, 1});
        q.submit([&](sycl::handler &h) {
             h.set_args(A.data());
             h.parallel_for(exe_range, kernel);
         }).wait();

        std::cout << A << std::endl;
    });
}

void example2_add_block2d() {
    std::cout << "\n=== Example 2: Add Block2D ===\n";

    auto ctx = create_configured_context();
    auto toto = const_tinytc_core_info_t();

    execute_with_error_handling([&]() {
        // Initialize tensors
        matrix<float> A(32, 32, 1);
        matrix<float> B(32, 32, 2);
        matrix<float> C(32, 32, 0);

        const std::string code = R"TinyTL(
func @add_block2d(%A: memref<f32x32x32> {alignment=128},
                  %B: memref<f32x32x32> {alignment=128},        
                  %C: memref<f32x32x32> {alignment=128})
    attributes{subgroup_size=16,work_group_size=[16,1]} {
    $mat_t = coopmatrix<f32x16x16,matrix_acc>
    parallel {
        %0 = constant 3 : index
        %1 = constant 2 : index
        %2 = cooperative_matrix_load %A[%0,%1] : $mat_t
        %3 = cooperative_matrix_load %B[%0,%1] : $mat_t
        %4 = add %2, %3 : $mat_t
        ; lambda to apply element-wise exponential
        %5 = cooperative_matrix_apply (%i,%j,%v)=%4 -> coopmatrix<f32x16x16,matrix_acc> {
            %exp_v = native_exp %v : f32
            yield (%exp_v)
        }
        cooperative_matrix_store %5, %C[%0,%1]
    }
})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "add_block2d");

        auto exe_range = tinytc::get_execution_range(kernel, sycl::range<3u>{1, 1, 1});
        q.submit([&](sycl::handler &h) {
             h.set_args(A.data(), B.data(), C.data());
             h.parallel_for(exe_range, kernel);
         }).wait();

        std::cout << C << std::endl;
    });
}

void example3_tiling_remainder() {
    std::cout << "\n=== Example 3: Tiling with Remainder ===\n";

    auto ctx = create_configured_context();
    auto toto = const_tinytc_core_info_t();

    execute_with_error_handling([&]() {
        // Initialize tensors
        matrix<float> A(36, 32, 1);
        matrix<float> B(36, 32, 0);

        const std::string code = R"TinyTL(
func @tilling(%A: memref<f32x36x32> {alignment=128},
              %B: memref<f32x36x32> {alignment=128})        
     attributes{subgroup_size=16,work_group_size=[16,16]} {
    ; Define a type alias for our cooperative matrix (16x16 tile of floats)
    $mat_t = coopmatrix<f32x16x16,matrix_acc>
    
    ; Matrix dimensions: 36 rows x 32 columns
    %c0 = constant 0 : index      ; Start position (row and column)
    %c1 = constant 36 : index     ; Total number of rows
    %c11 = constant 32 : index    ; Total number of columns
    %c2 = constant 2.0 : $mat_t   ; Value to add to full tiles
    %c4 = constant 3.0 : $mat_t   ; Value to add to remainder tiles

    ; TILING CONCEPT:
    ; We process the 36x32 matrix in 16x16 tiles.
    ; - Full tiles: 2 complete tiles (rows 0-15, 16-31) 
    ; - Remainder: partial tile with only 4 rows (rows 32-35)
    ;
    ; foreach_tile iterates over the matrix in tile-sized chunks:
    ;   (%i,%j) = current tile start position (top-left corner: 0, 16, 32...)
    ;   (%ti,%tj) = THE REMAINDER - actual tile dimensions
    ;               ti = min(16, remaining_rows), tj = min(16, remaining_cols)
    ;               For full tiles: ti=16, tj=16
    ;               For last tile (rows 32-35): ti=4, tj=16
    ;
    ; The loop processes tiles from (0,0) to (36,32) in steps of (16,16)
    foreach_tile (%i,%j)=(%c0,%c0),(%c1,%c11) as (%ti,%tj)<=(16,16) {
        %c3 = constant 16 : index
        
        ; Check if we have a partial tile by comparing remainder to full size
        ; If ti < 16, we have a partial tile (remainder)
        ; For our 36x32 matrix: first two tiles have ti=16, last tile has ti=4
        %is_remainder = less_than %ti, %c3 : bool
        
        if %is_remainder {
            ; REMAINDER TILE: Handle the partial tile (last 4 rows)
            ; Load the tile normally
            %tile = cooperative_matrix_load %A[%i,%j] : $mat_t
            ; Add 3.0 to distinguish remainder processing
            %tile_final = add %tile, %c4 : $mat_t
            ; Use .rows_checked to safely store only the valid rows
            ; This prevents writing beyond the matrix boundary
            cooperative_matrix_store.rows_checked %tile_final, %B[%i,%j]
        } else {
            ; FULL TILE: Process complete 16x16 tiles
            ; Load with .rows_checked (though not strictly necessary for full tiles)
            %tile = cooperative_matrix_load.rows_checked %A[%i,%j] : $mat_t
            ; Add 2.0 to distinguish full tile processing
            %tile_final = add %tile, %c2 : $mat_t
            ; Regular store (no bounds checking needed)
            cooperative_matrix_store %tile_final, %B[%i,%j]
        }
    }
})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "tilling");

        const auto global_range = sycl::range<3u>{1, 3, 3};

        q.submit([&](sycl::handler &h) {
             h.set_args(A.data(), B.data());
             h.parallel_for(tinytc::get_global_size(global_range, sycl::range<3u>(1, 16, 16)),
                            kernel);
         }).wait();

        std::cout << A << std::endl;
        std::cout << B << std::endl;
    });
}

void example4_subview() {
    std::cout << "\n=== Example 4: Subview Operations ===\n";

    auto ctx = create_configured_context();
    auto toto = const_tinytc_core_info_t();

    execute_with_error_handling([&]() {
        // original size
        const int64_t size = 64;
        // Initialize tensors
        matrix<float> A(size, size, 1);
        matrix<float> B(size, size, 0);

        const std::string code = R"TinyTL(
func @tilling(%A: memref<f32x64x64,strided<1,64>> {alignment=128,shape_gcd=[4,4],stride_gcd=[1,16]},
              %B: memref<f32x64x64,strided<1,64>> {alignment=128,shape_gcd=[4,4],stride_gcd=[1,16]})
     attributes{subgroup_size=16,work_group_size=[16,16]} {
    ; alias
    $mat_t = coopmatrix<f32x16x16,matrix_acc>
    %c0 = constant 0 : index
    %m1 = constant 1.0 : $mat_t
    %c16 = constant 16 : index

    parallel {
    %sv = subview %A[%c0:%c16,%c0:%c16] : memref<f32x?x?,strided<1,?>>
    %tile = cooperative_matrix_load %sv[%c0,%c0] : $mat_t
    %tile_final = add %tile, %m1 : $mat_t
    cooperative_matrix_store %tile_final, %B[%c0,%c0]
    }
})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "tilling");

        const int64_t tile_size = 16;
        const std::size_t gr_size = 1 + (A.rows() - 1) / tile_size;
        const auto global_range = sycl::range<3u>{1, 1, 1};

        std::cout << "gr_size: " << gr_size << std::endl;

        q.submit([&](sycl::handler &h) {
             h.set_args(A.data(), B.data());
             h.parallel_for(tinytc::get_global_size(global_range, sycl::range<3u>(1, 16, 16)),
                            kernel);
         }).wait();

        std::cout << B << std::endl;
    });
}

void example5_dynamic_memref() {
    std::cout << "\n=== Example 5: Dynamic Memref with Dope Vectors ===\n";

    auto ctx = create_configured_context();
    auto toto = const_tinytc_core_info_t();

    execute_with_error_handling([&]() {
        // original size
        const int64_t size = 64;
        // Initialize tensors
        matrix<float> A(size, size, 1);
        matrix<float> B(size, size, 0);

        const std::string code = R"TinyTL(
func @tilling(%A: memref<f32x?x?,strided<1,?>> {alignment=128,shape_gcd=[4,4],stride_gcd=[1,16]},
              %B: memref<f32x?x?,strided<1,?>> {alignment=128,shape_gcd=[4,4],stride_gcd=[1,16]})        
     attributes{subgroup_size=16,work_group_size=[16,16]} {
    ; alias
    $mat_t = coopmatrix<f32x16x16,matrix_acc>
    %c0 = constant 0 : index
    %m1 = constant 1.0 : $mat_t 
    %c16 = constant 16 : index 

    %sv = subview %A[%c0:%c16,%c0:%c16] : memref<f32x?x?,strided<1,?>>

    parallel {
    %tile = cooperative_matrix_load %sv[%c0,%c0] : $mat_t
    %tile_final = add %tile, %m1 : $mat_t
    cooperative_matrix_store %tile_final, %B[%c0,%c0]
    }
})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "tilling");

        const int64_t tile_size = 16;
        const std::size_t gr_size = 1 + (A.rows() - 1) / tile_size;
        const auto global_range = sycl::range<3u>{1, 1, 1};

        std::cout << "gr_size: " << gr_size << std::endl;

        // When using dynamic memref (memref<f32x?x?>) instead of static sizes,
        // we need to pass extra information so the kernel knows the matrix dimensions.
        // This is called a "dope vector" - it's just the size and layout info.
        //
        // For our 2D matrix, we pass 7 arguments total per matrix:
        //   1. Pointer to the data (A.data())
        //   2. Number of rows (A_shape0)
        //   3. Number of columns (A_shape1)
        //   4. Stride - how many elements to skip to go to next row (A_stride1)
        //
        // Example: For a 64x64 matrix stored row-by-row:
        //   - shape0 = 64 (64 rows)
        //   - shape1 = 64 (64 columns)
        //   - stride1 = 64 (skip 64 elements to go from one row to the next)
        std::int64_t A_shape0 = size;
        std::int64_t A_shape1 = size;
        std::int64_t A_stride1 = A_shape0;

        std::int64_t B_shape0 = size;
        std::int64_t B_shape1 = size;
        std::int64_t B_stride1 = B_shape1;

        q.submit([&](sycl::handler &h) {
             h.set_args(A.data(), A_shape0, A_shape1, A_stride1, B.data(), B_shape0, B_shape1,
                        B_stride1);
             h.parallel_for(tinytc::get_global_size(global_range, sycl::range<3u>(1, 16, 16)),
                            kernel);
         }).wait();

        std::cout << B << std::endl;
    });
}

void example6_dynamic_memref_tiling() {
    std::cout << "\n=== Example 6: Dynamic Memref with Tiling ===\n";

    auto ctx = create_configured_context();
    auto toto = const_tinytc_core_info_t();

    execute_with_error_handling([&]() {
        // original size
        const int64_t size = 32;
        // Initialize tensors
        matrix<float> A(size, size, 1);
        matrix<float> B(size, size, 0);

        const std::string code = R"TinyTL(
func @tilling(%A: memref<f32x?x?,strided<1,?>> {alignment=128,shape_gcd=[4,4],stride_gcd=[1,16]},
              %B: memref<f32x?x?,strided<1,?>> {alignment=128,shape_gcd=[4,4],stride_gcd=[1,16]})        
     attributes{subgroup_size=16,work_group_size=[16,16]} {
    ; alias
    $mat_t = coopmatrix<f32x16x16,matrix_acc>
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %A_rows = size %A[0] : index
    %A_cols = size %A[1] : index
    %m2 = constant 2.0 : $mat_t 

    foreach_tile (%i,%j)=(%c0,%c0),(%A_rows,%A_cols) as (%ti,%tj)<=(16,16) {
        ; ti and tj are the actual tile sizes (handle remainders)
        %svA = subview %A[%i:%ti,%j:%tj] : memref<f32x?x?,strided<1,?>>
        %svB = subview %B[%i:%ti,%j:%tj] : memref<f32x?x?,strided<1,?>>
        %tile = cooperative_matrix_load %svA[%c0,%c0] : $mat_t
        %tile_final = add %tile, %m2 : $mat_t
        cooperative_matrix_store %tile_final, %svB[%c0,%c0]
    }

})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "tilling");

        const int64_t tile_size = 16;
        const std::size_t gr_size = 1 + (A.rows() - 1) / tile_size;
        const auto global_range = sycl::range<3u>{1, gr_size, gr_size};

        std::int64_t A_shape0 = size;
        std::int64_t A_shape1 = size;
        std::int64_t A_stride1 = A_shape0;

        std::int64_t B_shape0 = size;
        std::int64_t B_shape1 = size;
        std::int64_t B_stride1 = B_shape1;

        q.submit([&](sycl::handler &h) {
             h.set_args(A.data(), A_shape0, A_shape1, A_stride1, B.data(), B_shape0, B_shape1,
                        B_stride1);
             h.parallel_for(tinytc::get_global_size(global_range, sycl::range<3u>(1, 16, 16)),
                            kernel);
         }).wait();

        std::cout << B << std::endl;
    });
}

void example7_tiling_manual_workgroup() {
    std::cout << "\n=== Example 7: manual tilling using workgroup ===\n";

    auto ctx = create_configured_context();
    auto toto = const_tinytc_core_info_t();

    execute_with_error_handling([&]() {
        // original size
        const int64_t size = 32;
        // Initialize tensors
        matrix<float> A(size, size, 1);
        matrix<float> B(size, size, 0);

        // Fill quarters: top-left=0, top-right=1, bottom-left=2, bottom-right=3
        for(int i=0; i<size; i++) {
            for(int j=0; j<size; j++) {
            int quarter = (i >= size/2 ? 2 : 0) + (j >= size/2 ? 1 : 0);
            B(i,j) = quarter;
            }
        }

        std::cout << "Matrix B before:\n" << B << std::endl;

        std::array<float, 4> W = {1.0, 2.0, 3.0, 4.0};

        const std::string code = R"TinyTL(
func @foo(%A: memref<f32x32x32> {alignment=128},
          %B: memref<f32x32x32> {alignment=128})        
    attributes{subgroup_size=16,work_group_size=[16,16]} {
    ; alias
    $mat_t = coopmatrix<f32x16x16,matrix_acc>
    ; the mystery ....
    %gz = group_id.z : index
    ;gy 0 or 1 
    %gy = group_id.y : index
    ;gx 0 or 1
    %gx = group_id.x : index

    %c16 = constant 16 : index
    ; beginning of the workgroup 9 or 16
    %gx_begin = mul %gx, %c16 : index
    %gy_begin = mul %gy, %c16 : index

    parallel {
        %1 = cooperative_matrix_load %A[%gx_begin,%gy_begin] : $mat_t
        %2 = cooperative_matrix_load %B[%gx_begin,%gy_begin] : $mat_t
        %3 = add %1, %2 : $mat_t
        cooperative_matrix_store %3, %B[%gx_begin,%gy_begin]
    }

})TinyTL";

        // JIT compile program
        auto q = sycl::queue{};
        auto program = tinytc::parse_string(code, ctx.get());
        auto bundle = tinytc::create_kernel_bundle(q.get_context(), q.get_device(), program.get());
        auto kernel = tinytc::create_kernel(bundle, "foo");

        q.submit([&](sycl::handler &h) {
             h.set_args(A.data(), B.data());
             // from Carsten it z,y,x order
             // https://intel.github.io/tiny-tensor-compiler/api/sycl/cxxapi.html#tinytc-get-global-size-sycl-range-3u-const-sycl-range-3u-const
             h.parallel_for(tinytc::get_global_size(sycl::range<3u>{1, 2, 2}, sycl::range<3u>(1, 16, 16)),
                            kernel);
         }).wait();

        std::cout << "Matrix B after :\n" << B << std::endl;
    });
}

} // namespace sandbox
