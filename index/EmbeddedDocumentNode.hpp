#include <istream>
#include <ostream>
#include <torch/script.h>

class EmbeddedDocumentNode {
public:
  std::string path;
  torch::Tensor embedding;
  int id;

  EmbeddedDocumentNode(torch::Tensor embedding, int id, std::string path) {
    this->embedding = embedding;
    this->id = id;
    this->path = path;
  }

  EmbeddedDocumentNode() {}

  double calculateL2(const torch::Tensor& b) const;

  EmbeddedDocumentNode operator+(const EmbeddedDocumentNode& other) const {
    return EmbeddedDocumentNode(this->embedding + other.embedding, -1, "");
  }

  EmbeddedDocumentNode operator+(const torch::Tensor& other) const {
      return EmbeddedDocumentNode(this->embedding + other, this->id, this->path);
  }

  void serialize(std::ostream& out) const;
  friend void deserializeNode(EmbeddedDocumentNode* node, std::istream& in);
};

