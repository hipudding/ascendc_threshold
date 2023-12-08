function(BUILD_EXEC NAME)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../../${NAME}.cpp PROPERTIES LANGUAGE CCE)

    add_executable(${NAME}_npu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../${NAME}.cpp
    )

    target_compile_options(${NAME}_npu PRIVATE
        -O2
        -std=c++17
    )

    target_link_libraries(${NAME}_npu PRIVATE
        ascendc_kernels
        -Wl,--as-needed
    )

    # set_target_properties(${NAME}_npu PROPERTIES
    #     OUTPUT_NAME ${NAME}_npu
    #     RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
    # )
endfunction(BUILD_EXEC)

foreach(test_op ${TEST_OPERATOR})
  BUILD_EXEC(${test_op})
endforeach()