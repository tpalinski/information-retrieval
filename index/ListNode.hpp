#include <cstdlib>
#include <functional>
#include <iostream>
#include <ostream>
#include <tuple>
template <typename T> class ListNode {
protected:
  ListNode<T>* next;
  T* value;

public:
  ListNode(T* value) {
    this->next = nullptr;
    this->value = value;
  }

  ~ListNode() {
    delete this->value;
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
      delete this->next;
      this->next = nullptr;
      return;
    } else {
      ListNode* newNext = this->next->next;
      delete this->next;
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

  void deleteByFilter(std::function<bool(T*)> filter) {
    if(this->next == nullptr) {
      if (filter(this->value)){ 
        delete this;
      }
      return;
    }
    if (filter(this->next->value)) {
      return this->deleteAfter();
    } else {
      return this->next->deleteByFilter(filter);
    }
  }

  std::tuple<T, int> reduceAdd(T startingValue, int count = 0) {
    T nextVal = startingValue + *(this->value);
    int nextCount = count+1;
    if (this->next == nullptr) {
      return std::tuple<T, int>(nextVal, nextCount);
    } else {
      return this->next->reduceAdd(nextVal, nextCount);
    }
  }

  int count(int offset = 0) const {
    if (this->next == nullptr) {
      return offset+1;
    } else {
      return this->next->count(offset+1);
    }
  }

  template<typename U>
  friend std::ostream& operator<<(std::ostream& os, const ListNode<U>& node);
};


template<typename T>
std::ostream& operator<<(std::ostream& os, const ListNode<T>& l) {
  os << *(l.value) << ", ";
  if (l.next == nullptr) {
    os << std::endl;
    return os;
  }
  return os << *l.next;
}
