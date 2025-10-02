cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ImageSearch)

find_package(Torch REQUIRED)

add_executable(main 
  main.cpp
  index/FlatIVFIndex.cpp
  index/EmbeddedDocumentNode.cpp
)
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/index)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)
