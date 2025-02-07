enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(IntervalTreeTest IntervalTreeTest.cpp)
target_link_libraries(IntervalTreeTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(IntervalTreeTest)

add_executable(GridTest GridTest.cpp)
target_link_libraries(GridTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(GridTest)

add_executable(VolumeFractionTest VolumeFractionTest.cpp)
target_link_libraries(VolumeFractionTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(VolumeFractionTest)

add_executable(ComputeVolumeFracCircleTest ComputeVolFracCircleTest.cpp)
target_link_libraries(ComputeVolumeFracCircleTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(ComputeVolumeFracCircleTest)

add_executable(KDTreeTest KDTreeTest.cpp)
target_link_libraries(KDTreeTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(KDTreeTest)

add_executable(VolumeFractionCurvTest VolumeFractionCurvTest.cpp)
target_link_libraries(VolumeFractionCurvTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(VolumeFractionCurvTest)

add_executable(TorchWrapperTest TorchWrapperTest.cpp)
target_link_libraries(TorchWrapperTest VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(TorchWrapperTest)