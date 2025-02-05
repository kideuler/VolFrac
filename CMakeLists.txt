cmake_minimum_required(VERSION 3.10)

project(VolFracAI)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()
include(CTest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/googletest)
include(GoogleTest)
#find_package(GTest REQUIRED)

# Create a library from all .cpp and .hpp files
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
add_library(VolFrac SHARED ${SOURCES})

# Include directories
target_include_directories(VolFrac PUBLIC src)

# Add testing
add_subdirectory(tests)

# Add data
add_subdirectory(data)