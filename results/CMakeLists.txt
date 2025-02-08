add_executable(Accuracy_test Accuracy_test.cpp)
target_link_libraries(Accuracy_test PRIVATE VolFrac)

# Add custom command to run results with an argument
add_custom_target(results
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_test 0 && ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_test 1 && ${CMAKE_CURRENT_BINARY_DIR}/Accuracy_test 2
    DEPENDS Accuracy_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running tests to generate results"
)