#include "EmbeddedDocumentNodeList.hpp"

void EmbeddedDocumentNodeList::heapify(DocumentNodeMinHeap* out, const torch::Tensor& reference) {
  out->insert(*(this->value), reference);
  if (this->next != nullptr) {
    static_cast<EmbeddedDocumentNodeList*>(this->next)->heapify(out, reference);
  }
}

void EmbeddedDocumentNodeList::serialize(std::ostream& out) const {
  this->value->serialize(out);
  bool hasNext = (this->next != nullptr);
  out.write(reinterpret_cast<const char*>(&hasNext), sizeof(hasNext));
  if (hasNext) {
    static_cast<EmbeddedDocumentNodeList*>(this->next)->serialize(out);
  }
}
