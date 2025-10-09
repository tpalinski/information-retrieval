cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ImageSearch)

find_package(Torch REQUIRED)

add_executable(main 
  main.cpp
  index/FlatIVFIndex.cpp
  index/EmbeddedDocumentNode.cpp
  index/DocumentNodeMinHeap.cpp
  index/EmbeddedDocumentNodeList.cpp
  serializer/utils.cpp
  inference/inference.cpp
)

target_include_directories(main PRIVATE 
  ${PROJECT_SOURCE_DIR}/index 
  ${PROJECT_SOURCE_DIR}/serializer 
  ${PROJECT_SOURCE_DIR}/inference
  ${PROJECT_SOURCE_DIR}/includes
  /opt/homebrew/include
)

target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}")

find_library(ZMQ_LIB NAMES zmq PATHS /opt/homebrew/lib)
if(NOT ZMQ_LIB)
    message(FATAL_ERROR "ZeroMQ library not found in /opt/homebrew/lib")
endif()
target_link_libraries(main PRIVATE ${ZMQ_LIB})

set_property(TARGET main PROPERTY CXX_STANDARD 17)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES arm64)
endif()

set(CMAKE_BUILD_TYPE Debug)
