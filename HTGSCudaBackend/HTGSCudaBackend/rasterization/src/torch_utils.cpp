#include "torch_utils.h"
#include <torch/extension.h>
#include <functional>

std::function<char*(size_t N)> resize_function_wrapper(torch::Tensor& t) {
    auto lambda = [&t](const size_t N) {
        t.resize_({static_cast<long long>(N)});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}
