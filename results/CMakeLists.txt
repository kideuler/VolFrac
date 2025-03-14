add_executable(Accuracy_test Accuracy_test.cpp)
target_link_libraries(Accuracy_test PRIVATE VolFrac)

add_executable(Accuracy_line_test Accuracy_line_test.cpp)
target_link_libraries(Accuracy_line_test PRIVATE VolFrac)

add_executable(Timing_test Timing_test.cpp)
target_link_libraries(Timing_test PRIVATE VolFrac)

configure_file(${CMAKE_SOURCE_DIR}/models/model.dat 
               ${CMAKE_CURRENT_BINARY_DIR}/model.dat 
               COPYONLY)

# Add custom command to run results with an argument
add_custom_target(results
    COMMAND echo "Running Accuracy test on Ellipse"
    && ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_test 0 
    && echo "Running Accuracy test on Flower"
    && ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_test 1 
    && echo "Running Accuracy test on Petals"
    && ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_test 2
    && echo "Running Accuracy test on Line"
    && ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_line_test
    && echo "Running Timing test on 100 point Ellipse"
    && ${CMAKE_CURRENT_BINARY_DIR}/Timing_test
    && echo "Running Timing test on 1000 point Ellipse"
    && ${CMAKE_CURRENT_BINARY_DIR}/Timing_test 1000
    && echo "Running Timing test on 5000 point Ellipse"
    && ${CMAKE_CURRENT_BINARY_DIR}/Timing_test 5000
    DEPENDS Accuracy_test Accuracy_line_test Timing_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tests to generate results"
)