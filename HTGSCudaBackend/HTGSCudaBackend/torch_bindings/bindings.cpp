#include <torch/extension.h>
#include "filter3d_api.h"
#include "rasterization_api.h"

namespace filter3d_api = htgs::filter3d;
namespace rasterization_api = htgs::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 3d filter
    m.def("update_3d_filter_cuda", &filter3d_api::update_3d_filter_wrapper);
    // unified rasterization api
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("render", &rasterization_api::inference_wrapper);
    m.def("update_max_weights", &rasterization_api::update_max_weights_wrapper);
}
