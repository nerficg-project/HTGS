#pragma once

#include <torch/extension.h>
#include <functional>

std::function<char*(size_t N)> resize_function_wrapper(torch::Tensor& t);
