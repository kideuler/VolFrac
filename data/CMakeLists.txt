add_executable(CreateTrainingData CreateTrainingData.cpp)
target_link_libraries(CreateTrainingData PRIVATE VolFrac)

add_executable(CreateBezierTrainingData CreateBezierTrainingData.cpp)
target_link_libraries(CreateBezierTrainingData PRIVATE VolFrac)

# Add custom command to run CreateTrainingData with an argument
add_custom_target(data
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/CreateTrainingData 200000
    DEPENDS CreateTrainingData
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CreateTrainingData to generate training data"
)

