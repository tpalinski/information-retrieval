#include <torch/script.h>

class EmbeddedDocumentNode {
public:
  torch::Tensor* embedding;
  int id;

  EmbeddedDocumentNode(torch::Tensor* embedding, int id) {
    this->embedding = embedding;
    this->id = id;
  }

  ~EmbeddedDocumentNode() {
    free(this->embedding);
  }

  double calculateL2(torch::Tensor b) {
    torch::Tensor diff = *this->embedding - b;
    torch::Tensor norm = torch::norm(diff, 2);
    return norm.item<double>();
  }
};
