#include "DocumentNodeMinHeap.hpp"
#include <utility>

void DocumentNodeMinHeap::insert(EmbeddedDocumentNode element, torch::Tensor reference) {
  this->elements.push_back(element);
  int currentIndex = this->elements.size()-1;
  int parentIndex = this->getParentIndex(currentIndex);
  while (parentIndex != -1) {
    double parentDistance = this->elements[parentIndex].calculateL2(reference);
    double thisDistance = this->elements[currentIndex].calculateL2(reference);
    if (thisDistance<parentDistance) {
      std::swap(this->elements[currentIndex], this->elements[parentIndex]);
      currentIndex = parentIndex;
      parentIndex = this->getParentIndex(currentIndex);
    } else {
      break;
    }
  }
}
