find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(IntervalTreeTest IntervalTreeTest.cpp)
target_link_libraries(IntervalTreeTest PRIVATE VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(IntervalTreeTest)

add_executable(GridTest GridTest.cpp)
target_link_libraries(GridTest PRIVATE VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(GridTest)

add_executable(VolumeFractionTest VolumeFractionTest.cpp)
target_link_libraries(VolumeFractionTest PRIVATE VolFrac GTest::GTest GTest::Main)
gtest_discover_tests(VolumeFractionTest)