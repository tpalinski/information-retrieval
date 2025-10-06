#include <istream>
#include <ostream>
#include <torch/script.h>

void saveTensor(std::ostream& out, const torch::Tensor& t);
torch::Tensor loadTensor(std::istream& in);
