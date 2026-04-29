cmake_minimum_required(VERSION 3.16)
project(consumer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vsomeip3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(consumer src/consumer.cpp)
target_link_libraries(consumer PRIVATE vsomeip3 Threads::Threads)
