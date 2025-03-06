#include <torch/extension.h>
#include "fast_inference_api.h"

namespace rasterization_api = htgs::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("benchmark", &rasterization_api::fast_inference_wrapper);
}
