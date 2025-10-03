#include "torch/script.h"
#include "EmbeddedDocumentNode.hpp"

torch::Tensor operator+(const torch::Tensor& t, const EmbeddedDocumentNode& node) {
    return t + node.embedding;
}


