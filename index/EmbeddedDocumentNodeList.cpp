#include "EmbeddedDocumentNodeList.hpp"

void EmbeddedDocumentNodeList::heapify(DocumentNodeMinHeap* out, const torch::Tensor& reference) {
  out->insert(*(this->value), reference);
  if (this->next != nullptr) {
    static_cast<EmbeddedDocumentNodeList*>(this->next)->heapify(out, reference);
  }
}
