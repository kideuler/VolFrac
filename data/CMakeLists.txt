add_executable(CreateTrainingData CreateTrainingData.cpp)
target_link_libraries(CreateTrainingData PRIVATE VolFrac)

# Add custom command to run CreateTrainingData with an argument
add_custom_target(data
    COMMAND CreateTrainingData ${ARG}
    DEPENDS CreateTrainingData
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CreateTrainingData to generate training data"
)

