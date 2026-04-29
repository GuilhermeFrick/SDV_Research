cmake_minimum_required(VERSION 3.16)
project(imu_service CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(vsomeip3 REQUIRED)
find_package(Threads REQUIRED)

add_executable(imu_service src/imu_service.cpp)
target_link_libraries(imu_service PRIVATE vsomeip3 Threads::Threads)
