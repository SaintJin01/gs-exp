// Feature-only alpha blending over the host 3DGS forward geometry buffers.
#include <torch/extension.h>
torch::Tensor blend_forward(torch::Tensor, torch::Tensor, torch::Tensor,
                            torch::Tensor, int64_t, int64_t, int64_t, double);
torch::Tensor blend_backward(torch::Tensor, torch::Tensor, torch::Tensor,
                             torch::Tensor, int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &blend_forward);
    m.def("backward", &blend_backward);
}
