#pragma once

#include <iostream>
#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>

namespace sandbox {

// Helper function to create and configure a compiler context
inline auto create_configured_context() {
    auto ctx = tinytc::create_compiler_context();
    tinytc::set_error_reporter(ctx.get(), 
        [](char const *what, const tinytc_location_t *, void *) {
            std::cerr << what << std::endl;
        });
    return ctx;
}

// Helper function to execute a kernel with error handling
template<typename Func>
void execute_with_error_handling(Func&& func) {
    try {
        func();
    } catch (tinytc::status const &st) {
        std::cerr << "Error (" << static_cast<int>(st) << "): " 
                  << tinytc::to_string(st) << std::endl;
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
    }
}

} // namespace sandbox
