#include <ostream>
#include <torch/script.h>

class EmbeddedDocumentNode {
public:
  torch::Tensor embedding;
  int id;

  EmbeddedDocumentNode(torch::Tensor embedding, int id) {
    this->embedding = embedding;
    this->id = id;
  }

  double calculateL2(const torch::Tensor& b) const;

  EmbeddedDocumentNode operator+(const EmbeddedDocumentNode& other) const {
    return EmbeddedDocumentNode(this->embedding + other.embedding, -1);
  }

  EmbeddedDocumentNode operator+(const torch::Tensor& other) const {
      return EmbeddedDocumentNode(this->embedding + other, this->id);
  }

  void serialize(std::ostream& out) const;
};

