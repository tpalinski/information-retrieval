#include <cstdlib>
#include <iostream>
#include "index/FlatFileMap.hpp"
#include "index/ListNode.hpp"
#include <torch/script.h>

using namespace std;

int main() {
  FlatFileMap<torch::Tensor, int>* map = new FlatFileMap<torch::Tensor, int>(3);
  torch::Tensor firstKey = torch::ones(2048) * 0.7;
  torch::Tensor secondKey = torch::ones(2048);
  int* val = new int();
  int* another = new int();
  int* third = new int();
  *val = 42;
  *another = 300;
  *third = 2137;
  map->put(firstKey, val);
  map->put(firstKey, another);
  map->put(secondKey, third);
  delete(map);
}
