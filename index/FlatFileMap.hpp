#include <torch/script.h>
#include <vector>
#include "EmbeddedDocumentNodeList.hpp"
#include "../serializer/utils.hpp"

template <typename T>
inline bool isEqual(const T& a, const T& b) {
    return a == b;
}

template <>
inline bool isEqual<torch::Tensor>(const torch::Tensor& a, const torch::Tensor& b) {
    return a.equal(b);
}


template <typename T, typename V> class FlatFileMap {
private: 
  int domainSize;
  int lastDictionaryKey = 0;
  V** hashedArray;
  T* dictionaryArray;

  int getKeyIndex(T key) const {
    for (int i = 0; i<this->lastDictionaryKey; i++) {
      if (isEqual(this->dictionaryArray[i], key)) {
        return i;
      }
    }
    return -1;
  }

public:
  void put(T key, V* value) {
    int keyIndex = this->getKeyIndex(key);
    if (keyIndex == -1 && lastDictionaryKey<domainSize) {
      this->dictionaryArray[lastDictionaryKey] = key;
      this->hashedArray[lastDictionaryKey] = value;
      this->lastDictionaryKey++;
    } else {
      this->hashedArray[keyIndex] = value;
    }
  }

  V* get(T key) const {
    int keyIndex = this->getKeyIndex(key);
    if (keyIndex == -1) {
      return nullptr;
    }
    return hashedArray[keyIndex];
  }

  inline bool exists(T key) const {
    return this->getKeyIndex(key) != -1;
  }

  inline std::vector<T> keys() const {
    return std::vector<T>(this->dictionaryArray, this->dictionaryArray+this->domainSize);
  }

  template<typename X = T, typename Y = V>
  typename std::enable_if<
    std::is_same<X, torch::Tensor>::value &&
    std::is_same<Y, EmbeddedDocumentNodeList>::value,
    void
  >::type
  serialize(std::ostream& out) {
    out.write(reinterpret_cast<const char*>(&domainSize), sizeof(domainSize));
    out.write(reinterpret_cast<const char*>(&lastDictionaryKey), sizeof(lastDictionaryKey));
    for (int i = 0; i < lastDictionaryKey; i++) {
      saveTensor(out, dictionaryArray[i]);
      hashedArray[i]->serialize(out);
    }
  }

  FlatFileMap(int domainSize) {
    this->domainSize = domainSize;
    this->hashedArray = new V*[domainSize];
    this->dictionaryArray = new T[domainSize];
  }

  FlatFileMap() {}

  ~FlatFileMap() {
    delete[] this->hashedArray;
    delete[] this->dictionaryArray;
  }

template <typename X, typename Y>
friend typename std::enable_if<
  std::is_same<X, torch::Tensor>::value &&
  std::is_same<Y, EmbeddedDocumentNodeList>::value,
  void
>::type
    deserializeMap(FlatFileMap<X, Y>* obj, std::istream& in);
};


template<typename T, typename V>
typename std::enable_if<
  std::is_same<T, torch::Tensor>::value &&
  std::is_same<V, EmbeddedDocumentNodeList>::value,
  void
>::type
deserializeMap(FlatFileMap<T, V>* obj, std::istream& in) {
  in.read(reinterpret_cast<char*>(&obj->domainSize), sizeof(obj->domainSize));
  in.read(reinterpret_cast<char*>(&obj->lastDictionaryKey), sizeof(obj->lastDictionaryKey));
  obj->dictionaryArray = new T[obj->lastDictionaryKey];
  obj->hashedArray = new V*[obj->lastDictionaryKey];
  for (int i = 0; i < obj->lastDictionaryKey; i++) {
    obj->dictionaryArray[i] = loadTensor(in);
    obj->hashedArray[i] = new EmbeddedDocumentNodeList();
    deserializeList(obj->hashedArray[i], in);
  }
}

