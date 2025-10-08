#include "EmbeddedDocumentNode.hpp"
#include "../serializer/utils.hpp"
#include <string>

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
  int pathLength = this->path.size();
  out.write(reinterpret_cast<const char*>(&pathLength), sizeof(pathLength));
  out.write(this->path.data(), pathLength);
  saveTensor(out, this->embedding);
}


void deserializeNode(EmbeddedDocumentNode* node, std::istream& in) {
  in.read(reinterpret_cast<char*>(&node->id), sizeof(node->id));
  int pathLength;
  in.read(reinterpret_cast<char*>(&pathLength), sizeof(pathLength));
  node->path = std::string(pathLength, '\0');
  in.read(&(node->path[0]), pathLength);
  node->embedding = loadTensor(in);
}
