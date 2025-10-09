#include "utils.hpp"
#include "../includes/msgpack.hpp"

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

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64Encode(const std::string &bytes) {
  std::string out;
  int val = 0, valb = -6;
  for (uint8_t c : bytes) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(base64_chars[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) out.push_back(base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

torch::Tensor unpackMsgpackTensor(const char* data, size_t size) {
  msgpack::object_handle oh = msgpack::unpack(data, size);
  msgpack::object obj = oh.get();
  std::map<std::string, msgpack::object> m;
  obj.convert(m);
  std::vector<int64_t> shape;
  std::string dtype;
  std::string raw_data;
  m["shape"].convert(shape);
  m["dtype"].convert(dtype);
  m["data"].convert(raw_data);
  if (dtype != "float32") {
    throw std::runtime_error("Expected float32 dtype");
  }
  if (shape.size() != 1) {
    throw std::runtime_error("Expected 1D tensor");
  }
  int64_t length = shape[0];
  if (raw_data.size() != static_cast<size_t>(length * sizeof(float))) {
    throw std::runtime_error("Size mismatch between shape and data bytes");
  }
  return torch::from_blob(
    (void*)raw_data.data(),
    {length},
    torch::kFloat32
  ).clone();
}
