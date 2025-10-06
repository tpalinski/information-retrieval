#include "ListNode.hpp"
#include "DocumentNodeMinHeap.hpp"
#include <istream>
#include <ostream>

class EmbeddedDocumentNodeList : public ListNode<EmbeddedDocumentNode> {
public:
  EmbeddedDocumentNodeList(EmbeddedDocumentNode* value) : ListNode<EmbeddedDocumentNode>(value) {}
  EmbeddedDocumentNodeList(): ListNode<EmbeddedDocumentNode>() {};
  void heapify(DocumentNodeMinHeap* out, const torch::Tensor& reference);
  void serialize(std::ostream& out) const;
  friend void deserializeList(EmbeddedDocumentNodeList* list, std::istream& in);
};
