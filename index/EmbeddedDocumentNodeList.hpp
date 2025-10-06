#include "ListNode.hpp"
#include "DocumentNodeMinHeap.hpp"

class EmbeddedDocumentNodeList : public ListNode<EmbeddedDocumentNode> {
public:
  EmbeddedDocumentNodeList(EmbeddedDocumentNode* value) : ListNode<EmbeddedDocumentNode>(value) {}
  void heapify(DocumentNodeMinHeap* out, const torch::Tensor& reference);
};
