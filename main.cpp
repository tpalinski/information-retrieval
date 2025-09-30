#include <cstdlib>
#include <iostream>
#include "FlatFileMap.hpp"
#include "ListNode.hpp"

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
  val = new int();
  another = new int();
  int* third = new int();
  *val = 42;
  *another = 300;
  *third = 3;
  ListNode<int>* head = new ListNode<int>(val);
  ListNode<int>* second = new ListNode<int>(another);
  ListNode<int>* eeee = new ListNode<int>(third);
  head->insertAfter(second);
  head->append(eeee);
  cout << "head value" << *(head->get());
  return 0;
}
