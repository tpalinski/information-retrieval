#include <string>
#include <vector>
#include "../index/FlatIVFIndex.hpp"

std::vector<std::string> getImages(const FlatIVFIndex& index, std::string query, int nresults = 10, int nprobe = 5);
FlatIVFIndex trainIndex(std::string datasetPath, std::string outPath, int ncells);
