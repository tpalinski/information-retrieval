#include <ostream>
#include <torch/script.h>

void saveTensor(std::ostream& out, const torch::Tensor& t) {
  auto data = t.contiguous();
  auto sizes = data.sizes();
  int64_t ndim = sizes.size();
  out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
  out.write(reinterpret_cast<const char*>(sizes.data()), ndim * sizeof(int64_t));
  int64_t numel = data.numel();
  out.write(reinterpret_cast<const char*>(data.data_ptr<float>()), numel * sizeof(float));
}
