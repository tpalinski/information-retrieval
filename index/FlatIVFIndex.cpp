#include "FlatIVFIndex.hpp"
#include <cstring>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>
#include <algorithm>

double calculateL2Distance(const torch::Tensor& a, const torch::Tensor& b) {
  torch::Tensor diff = a - b;
  return torch::norm(diff, 2).item<double>();
}

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
    double distance = calculateL2Distance(vector, cluster);
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
    EmbeddedDocumentNodeList* list = this->map->get(key);
    EmbeddedDocumentNode* node = new EmbeddedDocumentNode(tensor, id);
    EmbeddedDocumentNodeList* listNode = new EmbeddedDocumentNodeList(node);
    list->append(listNode);
  } else {
    EmbeddedDocumentNode* node = new EmbeddedDocumentNode(tensor, id);
    EmbeddedDocumentNodeList* listNode = new EmbeddedDocumentNodeList(node);
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
    this->map = new FlatFileMap<torch::Tensor, EmbeddedDocumentNodeList>(clusters);
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
        int newClusterIndex = chooseRandomIndices(tensors.size(), 1)[0];
        newCentroids[i] = tensors[newClusterIndex];
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
      std::cout << "Finished training, elements per centroid:" << std::endl;
      for (auto centroid : centroids) {
        auto cluster = this->map->get(centroid);
        std::cout << cluster->count() << std::endl;
      }
    }
  }
}

void FlatIVFIndex::train(std::vector<torch::Tensor>& tensors, int ncells) {
  this->runKMeans(tensors, ncells);
  this->isTrained = true;
}


std::vector<EmbeddedDocumentNode> FlatIVFIndex::find(const torch::Tensor& target, int nprobe, int nresults) const {
  std::vector<torch::Tensor> centroids = this->map->keys();
  std::nth_element(centroids.begin(), centroids.begin()+nprobe, centroids.end(), [target](const torch::Tensor& a, const torch::Tensor& b) {
    double aDistance = calculateL2Distance(a, target);
    double bDistance = calculateL2Distance(b, target);
    return aDistance < bDistance;
  });
  std::vector<EmbeddedDocumentNode> results;
  for (int i = 0; i<nprobe; i++) {
    EmbeddedDocumentNodeList* cluster = this->map->get(centroids[i]);
    DocumentNodeMinHeap* sorted = new DocumentNodeMinHeap();
    cluster->heapify(sorted, target);
    std::vector<EmbeddedDocumentNode> topResults = sorted->getTop(nresults);
    results.insert(results.end(), std::make_move_iterator(topResults.begin()), std::make_move_iterator(topResults.end()));
    delete sorted;
  }
  std::nth_element(results.begin(), results.begin()+nresults, results.end(), [target](const EmbeddedDocumentNode& a, const EmbeddedDocumentNode& b) {
    double aDistance = a.calculateL2(target);
    double bDistance = b.calculateL2(target);
    return aDistance < bDistance;
  });
  int resCount = std::min(nresults, (int) results.size());
  return std::vector(results.begin(), results.begin()+resCount);
}

void saveIndex(const FlatIVFIndex& index, std::string location) {
  return;
}

void loadIndex(FlatIVFIndex* index, std::string location) {
  return;
}
