include(CommonOptions)

find_package(SYCL REQUIRED)

# Build the unified sandbox executable
add_executable(sandbox main.cpp examples.cpp matrix.hpp common.hpp examples.hpp)
add_sycl_to_target(TARGET sandbox SOURCES main.cpp examples.cpp)
target_link_libraries(sandbox PRIVATE tinytc tinytc_sycl argparser)
