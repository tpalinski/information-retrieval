#include <algorithm>
#include <vector>
#include "EmbeddedDocumentNode.hpp"

class DocumentNodeMinHeap {
private:
  std::vector<EmbeddedDocumentNode> elements;

  inline int getParentIndex(int nodeIndex) {
    return (nodeIndex-1)/2;
  }
  inline int getLeftChildIndex(int nodeIndex) {
    return nodeIndex*2 + 1;
  }
  inline int getRightChildindex(int nodeIndex) {
    return nodeIndex*2 + 2;
  }

public:
  DocumentNodeMinHeap() {
    this->elements = std::vector<EmbeddedDocumentNode>();
  }

  DocumentNodeMinHeap(int capacity) {
    this->elements = std::vector<EmbeddedDocumentNode>();
    this->elements.reserve(capacity);
  }

  inline std::vector<EmbeddedDocumentNode> get() {
    return std::vector<EmbeddedDocumentNode>(this->elements.begin(), this->elements.end());
  }

  inline std::vector<EmbeddedDocumentNode> getTop(int n) {
    int resCount = std::min(n, (int)this->elements.size());
    return std::vector<EmbeddedDocumentNode>(this->elements.begin(), this->elements.begin()+resCount);
  }

  void insert(EmbeddedDocumentNode element, torch::Tensor reference);
};
