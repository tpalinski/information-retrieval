#include "EmbeddedDocumentNode.hpp"
#include "../serializer/utils.hpp"

torch::Tensor operator+(const torch::Tensor& t, const EmbeddedDocumentNode& node) {
    return t + node.embedding;
}


double EmbeddedDocumentNode::calculateL2(const torch::Tensor& b) const {
  torch::Tensor diff = this->embedding - b;
  torch::Tensor norm = torch::norm(diff, 2);
  return norm.item<double>();
}


void EmbeddedDocumentNode::serialize(std::ostream& out) const {
  out.write(reinterpret_cast<const char*>(&this->id), sizeof(this->id));
  saveTensor(out, this->embedding);
}
