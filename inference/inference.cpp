#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <torch/script.h>
#include <vector>
#include "inference.hpp"
#include "../includes/msgpack.hpp"

#define EMBEDDING_SIZE 512
#define TEXT_MESSAGE "text"
#define IMAGE_MESSAGE "image"


void sendRequest(zmq::socket_t& sock, const std::string& type, const std::string& payload_str) {
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> pk(&sbuf);
  pk.pack_map(2);
  pk.pack("type"); pk.pack(type);
  pk.pack("payload"); pk.pack(payload_str);
  zmq::message_t msg(sbuf.size());
  memcpy(msg.data(), sbuf.data(), sbuf.size());
  sock.send(msg, zmq::send_flags::none);
}

torch::Tensor receive_embedding(zmq::socket_t& sock) {
  zmq::message_t reply;
  sock.recv(reply, zmq::recv_flags::none);
  try {
    return unpackMsgpackTensor(static_cast<const char*>(reply.data()), reply.size());
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to unpack tensor: ") + e.what());
  }
}

torch::Tensor generateEmbedding(std::string query, zmq::socket_t& sock) {
  sendRequest(sock, TEXT_MESSAGE, query);
  return receive_embedding(sock);
}

std::vector<std::string> getImages(const FlatIVFIndex& index, zmq::socket_t& sock, std::string query, int nresults, int nprobe) {
  torch::Tensor embedding = generateEmbedding(query, sock);
  std::vector<EmbeddedDocumentNode> searchResults = index.find(embedding, nprobe, nresults);
  std::vector<std::string> results(searchResults.size());
  std::transform(searchResults.begin(), searchResults.end(), results.begin(), [](EmbeddedDocumentNode e) {
    return e.path;
  });
  return results;
}

torch::Tensor getImageEmbedding(std::string path, zmq::socket_t& sock) {
  sendRequest(sock, IMAGE_MESSAGE, path);
  return receive_embedding(sock);
}

FlatIVFIndex trainIndex(std::string datasetPath, std::string outPath, zmq::socket_t& sock, int ncells) {
  std::vector<std::string> paths;
  for (const auto& e : std::filesystem::directory_iterator(datasetPath)) {
    if (e.is_regular_file()) {
      auto extension = e.path().extension().string();
      if (extension == ".jpg" || extension == ".jpeg") {
        paths.push_back(e.path());
      }
    }
  }
  std::vector<torch::Tensor> embeddings;
  embeddings.reserve(paths.size());
  int i = 1;
  for (const std::string& path : paths) {
    std::cout << "Generating embedding for image: " << i <<"/" << paths.size() << std::endl;
    torch::Tensor embedding = getImageEmbedding(path, sock);
    embeddings.push_back(embedding);
    i++;
  }
  int dims = embeddings[0].size(0);
  FlatIVFIndex index(dims);
  index.train(embeddings, paths, ncells);
  saveIndex(index, outPath);
  return index;
}
