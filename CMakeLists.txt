cmake_minimum_required(VERSION 3.24)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_STRING)
string(STRIP "${PROJECT_VERSION_STRING}" PROJECT_VERSION_STRING)

# Extract version components
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${PROJECT_VERSION_STRING}")
set(PROJECT_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(PROJECT_VERSION_MINOR "${CMAKE_MATCH_2}")
set(PROJECT_VERSION_PATCH "${CMAKE_MATCH_3}")

# Configure project with version
project(VolFrac
        VERSION "${PROJECT_VERSION_STRING}"
        DESCRIPTION "Volume Fraction Calculation with AI"
        LANGUAGES CXX)

# Output version information
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VERSION}")

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# make tests optional
option(BUILD_TESTS "Build tests" ON)

# Option for building documentation
option(BUILD_DOC "Build documentation with Doxygen" OFF)

# Find OpenMP
option(USE_OPENMP "Use OpenMP for parallel execution" OFF)
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

# Set version for shared library
set_target_properties(VolFrac PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Link OpenMP
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(VolFrac PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(VolFrac PUBLIC USE_OPENMP)
    message(STATUS "OpenMP found and enabled")
endif()

# Include directories
target_include_directories(VolFrac PUBLIC src models results)

# Add testing
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Add data
add_subdirectory(data)

# Add results
add_subdirectory(results)

# Configure Doxygen documentation if BUILD_DOC is ON
if(BUILD_DOC)
    # Find Doxygen package
    find_package(Doxygen REQUIRED)
    
    # Check if Doxygen was found
    if(DOXYGEN_FOUND)
        message(STATUS "Doxygen found. Documentation will be generated.")
        
        # Set input and output files for Doxygen
        set(DOXYGEN_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/models")
        set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs")
        set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/html/index.html")
        
        # Create a Doxyfile with CMake variables
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            @ONLY
        )

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/mainpage.md
            ${CMAKE_CURRENT_BINARY_DIR}/docs/mainpage.md
            COPYONLY
        )
        
        # Add a custom target for building documentation
        add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        # Make the documentation part of the ALL target
        add_custom_target(doc_all ALL DEPENDS doc)
        
        # Install documentation
        install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html
                DESTINATION share/doc/${PROJECT_NAME}
                OPTIONAL)
                
        message(STATUS "Doxygen documentation will be generated in: ${DOXYGEN_OUTPUT_DIR}")
    else()
        message(WARNING "Doxygen not found. Documentation will not be generated.")
    endif()
endif()