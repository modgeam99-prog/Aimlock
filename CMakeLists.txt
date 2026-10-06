cmake_minimum_required(VERSION 3.18)
project(aimlock CXX)

add_library(aimlock SHARED aimlock_android.cpp)

target_compile_options(aimlock PRIVATE
    -O2
    -fPIC
    -fvisibility=hidden
    -std=c++17
)

target_link_libraries(aimlock log android)
