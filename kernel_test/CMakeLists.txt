INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../ascendc_operators)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../ascendc_operators/include)

set(TEST_OPERATOR 
    test_threshold)

add_subdirectory(cmake/cpu)
add_subdirectory(cmake/npu)

