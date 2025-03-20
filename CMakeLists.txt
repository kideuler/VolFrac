cmake_minimum_required(VERSION 3.24)

project(VolFracAI)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

include(CTest)

# Create a library from all .cpp and .hpp files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/models/*.hpp")
add_library(VolFrac SHARED ${SOURCES})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(VolFrac PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
  )
endif()

# Link OpenMP
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(VolFrac PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(VolFrac PUBLIC USE_OPENMP)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found. Parallel execution will be disabled.")
endif()

# Include directories
target_include_directories(VolFrac PUBLIC src models results)

# Add testing
add_subdirectory(tests)

# Add data
add_subdirectory(data)

# Add results
add_subdirectory(results)