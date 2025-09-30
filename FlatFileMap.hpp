template <typename T, typename V> class FlatFileMap {
private: 
  int domainSize;
  int lastDictionaryKey = 0;
  V** hashedArray;
  T* dictionaryArray;

  int getKeyIndex(T key) {
    for (int i = 0; i<this->domainSize; i++) {
      if (this->dictionaryArray[i] == key) {
        return i;
      }
    }
    return -1;
  }

public:
  void put(T key, V* value) {
    int keyIndex = this->getKeyIndex(key);
    if (keyIndex == -1 && lastDictionaryKey<domainSize-1) {
      this->dictionaryArray[lastDictionaryKey] = key;
      this->hashedArray[lastDictionaryKey] = value;
      this->lastDictionaryKey++;
    } else {
      this->hashedArray[keyIndex] = value;
    }
  }

  V* get(T key) {
    int keyIndex = this->getKeyIndex(key);
    if (keyIndex == -1) {
      return nullptr;
    }
    return hashedArray[keyIndex];
  }

  bool exists(T key) {
    return this->getKeyIndex(key) != -1;
  }

  FlatFileMap(int domainSize) {
    this->domainSize = domainSize;
    this->hashedArray = new V*[domainSize];
    this->dictionaryArray = new T[domainSize];
  }

  ~FlatFileMap() {
    for (int i = 0; i<this->domainSize; i++) {
      delete(hashedArray[i]);
    }
    delete(this->hashedArray);
    delete(this->dictionaryArray);
  }

};
