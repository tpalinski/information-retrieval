#include <cstdlib>
template <typename T> class ListNode {
private:
  ListNode<T>* next;
  T* value;

public:
  ListNode(T* value) {
    this->next = nullptr;
    this->value = value;
  }

  T* get() {
    return this->value;
  }

  void insertAfter(ListNode* newNext) {
    if (this->next == nullptr) {
      this->next = newNext;
    } else {
      newNext->next = this->next;
      this->next = newNext;
    }
  }

  void deleteAfter() {
    if (this->next == nullptr) {
      return;
    }
    if (this->next->next == nullptr) {
      free(this->next);
      this->next == nullptr;
      return;
    } else {
      ListNode* newNext = this->next->next;
      free(this->next);
      this->next = newNext;
    }
  }

  void append(ListNode* node) {
    if (this->next == nullptr) {
      this->next = node;
      return;
    }
    this->next->append(node);
  }
};
