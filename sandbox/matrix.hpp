#pragma once

#include <cstdint>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

namespace sandbox {

// Forward declaration for global queue
extern const sycl::queue q;

template <typename T>
class matrix {
public:
    using value_type = T;
    constexpr static T poison = T(-1);

    matrix(std::int64_t rows, std::int64_t cols, T initial_value = poison)
        : rows_{rows}, cols_{cols}, alloc_(q), data_(rows * cols, initial_value, alloc_) {}

    inline auto rows() const -> std::int64_t { return rows_; }
    inline auto cols() const -> std::int64_t { return cols_; }
    inline auto bytes() const -> std::size_t { return rows_ * cols_ * sizeof(T); }

    inline auto data() -> T * { return data_.data(); }
    inline auto data() const -> T const * { return data_.data(); }

    inline auto operator()(std::int64_t i, std::int64_t j) -> T & { 
        return data_[i + j * rows_]; 
    }
    
    inline auto operator()(std::int64_t i, std::int64_t j) const -> T const & {
        return data_[i + j * rows_];
    }

    void print(std::ostream &os) const {
        for (std::int64_t i = 0; i < rows_; ++i) {
            for (std::int64_t j = 0; j < cols_; ++j) {
                os << operator()(i, j) << " ";
            }
            os << "\n";
        }
    }

private:
    std::int64_t rows_, cols_;
    sycl::usm_allocator<T, sycl::usm::alloc::shared> alloc_;
    std::vector<T, decltype(alloc_)> data_;
};



template<class T>
std::ostream &operator<<(std::ostream &os, const matrix<T> &mat) {
    mat.print(os);
    return os;
}

} // namespace sandbox
