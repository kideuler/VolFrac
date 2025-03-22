add_executable(CreateTrainingData_Curve CreateTrainingData_Curve.cpp)
target_link_libraries(CreateTrainingData_Curve PRIVATE VolFrac)

add_executable(CreateTrainingData_Plane CreateTrainingData_Plane.cpp)
target_link_libraries(CreateTrainingData_Plane PRIVATE VolFrac)

add_executable(CreateBezierTrainingData CreateBezierTrainingData.cpp)
target_link_libraries(CreateBezierTrainingData PRIVATE VolFrac)

# Add custom command to run CreateTrainingData with an argument
add_custom_target(data
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/CreateTrainingData_Curve 100000
    && ${CMAKE_CURRENT_BINARY_DIR}/CreateTrainingData_Plane 50000
    DEPENDS CreateTrainingData_Curve CreateBezierTrainingData
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running data generation programs to create training data"
)