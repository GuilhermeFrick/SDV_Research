cmake_minimum_required(VERSION 3.16)
project(gps_service CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vsomeip3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(gps_service src/gps_service.cpp)
target_link_libraries(gps_service PRIVATE vsomeip3 Threads::Threads)
