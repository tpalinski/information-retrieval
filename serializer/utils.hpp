#include <ostream>
#include <torch/script.h>

void saveTensor(std::ostream& out, const torch::Tensor& t);
