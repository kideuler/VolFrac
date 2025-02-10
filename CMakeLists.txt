cmake_minimum_required(VERSION 3.24)

project(VolFracAI)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(CTest)

# Create a library from all .cpp and .hpp files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.hpp")
add_library(VolFrac SHARED ${SOURCES})

# add torch if USE_TORCH is set to true
if(USE_TORCH)
  find_package(Torch REQUIRED)
  target_link_libraries(VolFrac PUBLIC ${TORCH_LIBRARIES})
endif()

# Include directories
target_include_directories(VolFrac PUBLIC src models results)

# Add testing
add_subdirectory(tests)

# Add data
add_subdirectory(data)

# Add results
add_subdirectory(results)