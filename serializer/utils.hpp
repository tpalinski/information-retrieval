#include <istream>
#include <ostream>
#include <string>
#include <torch/script.h>

void saveTensor(std::ostream& out, const torch::Tensor& t);
torch::Tensor loadTensor(std::istream& in);
std::string base64Encode(const std::string &bytes);
