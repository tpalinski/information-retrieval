#include <string>
#include <vector>
#include "../index/FlatIVFIndex.hpp"
#include "../includes/zmq.hpp"

std::vector<std::string> getImages(const FlatIVFIndex& index, zmq::socket_t& sock, std::string query, int nresults = 10, int nprobe = 5);
FlatIVFIndex trainIndex(std::string datasetPath, std::string outPath, zmq::socket_t& sock, int ncells);
