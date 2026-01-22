#include "examples.hpp"
#include <iostream>
#include <string>
#include <map>

namespace sandbox {
const sycl::queue q{};
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [example_number]\n\n";
    std::cout << "Available examples:\n";
    std::cout << "  1 - Store Block2D: Demonstrates cooperative matrix store\n";
    std::cout << "  2 - Add Block2D: Demonstrates cooperative matrix operations with apply\n";
    std::cout << "  3 - Tiling with Remainder: Shows tiling with remainder checking\n";
    std::cout << "  4 - Subview Operations: Demonstrates subview functionality\n";
    std::cout << "  5 - Dynamic Memref: Shows dynamic memref with dope vectors\n";
    std::cout << "  6 - Dynamic Memref with Tiling: Shows dynamic memref with tiling\n";
    std::cout << "  7 - Manual Tiling with Workgroup Management\n";
    std::cout << "  all - Run all examples sequentially\n\n";
    std::cout << "If no argument is provided, runs all examples.\n";
}

int main(int argc, char *argv[]) {
    using namespace sandbox;
    
    // Map of example numbers to functions
    std::map<int, void(*)()> examples = {
        {1, example1_store_block2d},
        {2, example2_add_block2d},
        {3, example3_tiling_remainder},
        {4, example4_subview},
        {5, example5_dynamic_memref},
        {6, example6_dynamic_memref_tiling },
        {7, example7_tiling_manual_workgroup}

    };

    // If no arguments or "all", run all examples
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "all")) {
        std::cout << "Running all examples...\n";
        for (const auto& [num, func] : examples) {
            func();
        }
        return 0;
    }

    // Check for help flag
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    // Run specific example
    if (argc == 2) {
        try {
            int example_num = std::stoi(argv[1]);
            if (examples.find(example_num) != examples.end()) {
                examples[example_num]();
                return 0;
            } else {
                std::cerr << "Error: Invalid example number " << example_num << "\n\n";
                print_usage(argv[0]);
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid argument '" << argv[1] << "'\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Too many arguments
    std::cerr << "Error: Too many arguments\n\n";
    print_usage(argv[0]);
    return 1;
}
