#include <torch/script.h>

class EmbeddedDocumentNode {
public:
  torch::Tensor embedding;
  int id;

  EmbeddedDocumentNode(torch::Tensor embedding, int id) {
    this->embedding = embedding;
    this->id = id;
  }

  double calculateL2(torch::Tensor b) {
    torch::Tensor diff = this->embedding - b;
    torch::Tensor norm = torch::norm(diff, 2);
    return norm.item<double>();
  }

  EmbeddedDocumentNode operator+(const EmbeddedDocumentNode& other) const {
    return EmbeddedDocumentNode(this->embedding + other.embedding, -1);
  }

  EmbeddedDocumentNode operator+(const torch::Tensor& other) const {
      return EmbeddedDocumentNode(this->embedding + other, this->id);
  }
};

