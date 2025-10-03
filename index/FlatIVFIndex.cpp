#include "FlatIVFIndex.hpp"
#include <cstring>
#include <iostream>
#include <numeric>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>
#include <algorithm>

std::vector<int> chooseRandomIndices(int total, int nResults) {
  std::vector<int> indices(total);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(indices.begin(), indices.end(), gen);
  return std::vector(indices.begin(), indices.begin()+nResults);
}

torch::Tensor assignToCluster(std::vector<torch::Tensor> clusters, const torch::Tensor vector) {
  double minDistance = MAXFLOAT;
  torch::Tensor closest;
  for (torch::Tensor cluster : clusters) {
    torch::Tensor diff = (cluster - vector);
    double distance = torch::norm(diff, 2).item<double>();
    if (distance < minDistance) {
      closest = cluster;
      minDistance = distance;
    }
  }
  return closest;
}

void FlatIVFIndex::addTensorToMap(torch::Tensor& tensor, const torch::Tensor& key, int id) {
  if (this->map == nullptr) {
    return;
  }
  if (this->map->exists(key)) {
    ListNode<EmbeddedDocumentNode>* list = this->map->get(key);
    EmbeddedDocumentNode* node = new EmbeddedDocumentNode(tensor, id);
    ListNode<EmbeddedDocumentNode>* listNode = new ListNode<EmbeddedDocumentNode>(node);
    list->append(listNode);
  } else {
    EmbeddedDocumentNode* node = new EmbeddedDocumentNode(tensor, id);
    ListNode<EmbeddedDocumentNode>* listNode = new ListNode<EmbeddedDocumentNode>(node);
    this->map->put(key, listNode);
  }
}

void FlatIVFIndex::runKMeans(std::vector<torch::Tensor>& tensors, int clusters) {
  // Random intialization amongst the existing population
  int tensorCount = tensors.size();
  std::vector<torch::Tensor> centroids(clusters);
  std::vector<int> firstIndices = chooseRandomIndices(tensorCount, clusters);
  for (int i = 0; i<clusters; i++) {
    centroids[i] = tensors[firstIndices[i]];
  }
  bool converged = false;
  int iterations = 0;
  // Main loop
  while (!converged) {
    iterations++;
    std::cout << "Partitioning space. Iteration: " << iterations << std::endl;
    if(this->map != nullptr) {
      delete this->map;
    }
    std::cout << "Assigning to centroids" << std::endl;
    // assign to closest centroid
    this->map = new FlatFileMap<torch::Tensor, ListNode<EmbeddedDocumentNode>>(clusters);
    for (int i = 0; i<tensorCount; i++) {
      torch::Tensor closestCluster = assignToCluster(centroids, tensors[i]);
      this->addTensorToMap(tensors[i], closestCluster, i);
    } 
    std::cout << "Recalculating centroids" << std::endl;
    // recalculate new centroids
    std::vector<torch::Tensor> newCentroids(clusters);
    for (int i = 0; i<clusters; i++) {
      ListNode<EmbeddedDocumentNode>* cluster = this->map->get(centroids[i]);
      // possible that none of the og clusters were closest. 
      if (cluster == nullptr) {
        newCentroids[i] = centroids[i];
        continue;
      }
      std::tuple<EmbeddedDocumentNode, int> reduced = cluster->reduceAdd(EmbeddedDocumentNode(torch::zeros(this->dims), -1));
      torch::Tensor newMean = std::get<0>(reduced).embedding / (double)std::get<1>(reduced);
      newCentroids[i] = newMean;
    }
    // Stop check
    std::cout << "Checking convergence" << std::endl;
    converged = true;
    for (int i = 0; i<clusters; i++) {
      if (!newCentroids[i].equal(centroids[i])) {
        converged = false;
        break;
      }
    }
    centroids = newCentroids;
    if (converged) {
      std::cout << "Centroids: \n" << centroids << std::endl;
    }
  }
}

void FlatIVFIndex::train(std::vector<torch::Tensor>& tensors, int ncells) {
  this->runKMeans(tensors, ncells);
  this->isTrained = true;
}


const std::vector<int> FlatIVFIndex::find(const torch::Tensor& target, int nprobe, int nresults) {
  
}

void saveIndex(const FlatIVFIndex& index, std::string location) {
  return;
}

void loadIndex(FlatIVFIndex* index, std::string location) {
  return;
}
