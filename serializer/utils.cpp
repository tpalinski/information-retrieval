#include "utils.hpp"

void saveTensor(std::ostream& out, const torch::Tensor& t) {
  auto data = t.contiguous();
  auto sizes = data.sizes();
  int64_t ndim = sizes.size();
  out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
  out.write(reinterpret_cast<const char*>(sizes.data()), ndim * sizeof(int64_t));
  int64_t numel = data.numel();
  out.write(reinterpret_cast<const char*>(data.data_ptr<float>()), numel * sizeof(float));
}

torch::Tensor loadTensor(std::istream& in) {
  int64_t ndim;
  in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
  std::vector<int64_t> sizes(ndim);
  in.read(reinterpret_cast<char*>(sizes.data()), ndim * sizeof(int64_t));
  int64_t numel = 1;
  for (int64_t s : sizes) {
      numel *= s;
  }
  std::vector<float> buffer(numel);
  in.read(reinterpret_cast<char*>(buffer.data()), numel * sizeof(float));
  torch::Tensor t = torch::from_blob(buffer.data(), sizes, torch::kFloat).clone();
  return t;
}
