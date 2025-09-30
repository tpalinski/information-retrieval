#include <cstdlib>
#include <iostream>
#include "FlatFileMap.hpp"

using namespace std;

int main() {
  FlatFileMap<int, int>* map = new FlatFileMap<int, int>(3);
  int* val = new int();
  int* another = new int();
  *val = 42;
  *another = 300;
  map->put(2137, val);
  map->put(2138, another);
  delete(map);
  return 0;
}
