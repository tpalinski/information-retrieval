#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <torch/script.h>
#include <vector>
#include "includes/httplib.h"
#include "inference/inference.hpp"
#include "includes/json.hpp"

using namespace std;
using json = nlohmann::json;

#define INDEX_SAVE_PATH "data/index.bin"
#define DATASET_PATH "data/img/"
#define INDEX_CLUSTERS 250

int main() {
  torch::NoGradGuard gradGuard;
  httplib::Server server;

  FlatIVFIndex index; 

  if (std::filesystem::exists(INDEX_SAVE_PATH)) {
    cout << "Found trained index at " << INDEX_SAVE_PATH << endl;
    loadIndex(&index, INDEX_SAVE_PATH);
  } else {
    cout << "Could not find trained index, training... " << endl;
    index = trainIndex(DATASET_PATH, INDEX_SAVE_PATH, INDEX_CLUSTERS);
  }

  server.Post("/query",  [&index](const httplib::Request& req, httplib::Response& res){
    try {
      json input = json::parse(req.body);
      int nprobe = input.at("nprobe").get<int>();
      int nresults = input.at("nresults").get<int>();
      string query = input.at("query").get<string>();
      vector<string> results = getImages(index, query, nresults, nprobe);
      json output;
      output["images"] = json::array();
      for (string image : results) {
        ifstream file(image, std::ios::binary);
        string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        string encoded = base64Encode(buffer);
        output["images"].push_back({encoded});
      }
      res.set_content(output.dump(), "application/json");
    } catch(const std::exception e) {
      json error = {{"error", e.what()}};
      res.status = 400;
      res.set_content(error.dump(), "application/json");
    }
  });

  std::cout << "Server running at http://localhost:2137" << std::endl;
  server.listen("0.0.0.0", 2137);
}
