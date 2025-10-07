#include <cstdlib>
#include <torch/script.h>
#include "inference/inference.hpp"

using namespace std;

int main() {
  torch::NoGradGuard gradGuard;
}
