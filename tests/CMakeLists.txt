enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(IntervalTreeTest IntervalTreeTest.cpp)
target_link_libraries(IntervalTreeTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(IntervalTreeTest)

add_executable(GridTest GridTest.cpp)
target_link_libraries(GridTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(GridTest)

add_executable(VolumeFractionTest VolumeFractionTest.cpp)
target_link_libraries(VolumeFractionTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(VolumeFractionTest)

add_executable(ComputeVolumeFracCircleTest ComputeVolFracCircleTest.cpp)
target_link_libraries(ComputeVolumeFracCircleTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(ComputeVolumeFracCircleTest)

add_executable(KDTreeTest KDTreeTest.cpp)
target_link_libraries(KDTreeTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(KDTreeTest)

add_executable(VolumeFractionCurvTest VolumeFractionCurvTest.cpp)
target_link_libraries(VolumeFractionCurvTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(VolumeFractionCurvTest)

add_executable(MatrixVectorTest MatrixVectorTest.cpp)
target_link_libraries(MatrixVectorTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(MatrixVectorTest)

# Copy model.dat file to the test directory
configure_file(${CMAKE_SOURCE_DIR}/models/model.dat 
               ${CMAKE_CURRENT_BINARY_DIR}/model.dat 
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/models/model_planes.dat 
               ${CMAKE_CURRENT_BINARY_DIR}/model_planes.dat 
               COPYONLY)

add_executable(ModelTest ModelTest.cpp)
target_link_libraries(ModelTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(ModelTest)

add_executable(BezierTest BezierTest.cpp)
target_link_libraries(BezierTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(BezierTest)

add_executable(PolyVolFracTest PolyVolFracTest.cpp)
target_link_libraries(PolyVolFracTest GTest::GTest GTest::Main VolFrac)
gtest_discover_tests(PolyVolFracTest)