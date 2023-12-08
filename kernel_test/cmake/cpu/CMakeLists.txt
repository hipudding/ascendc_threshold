function(BUILD_EXEC NAME)
    add_executable(${NAME}_cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/../../${NAME}.cpp
    )

    target_compile_options(${NAME}_cpu PRIVATE
        -O2
        -std=c++17
    )

    target_link_libraries(${NAME}_cpu PRIVATE
        tikicpulib::${ASCEND_PRODUCT_TYPE}
        ascendcl
        ascendc_kernels_debug
    )

    target_compile_definitions(${NAME}_cpu PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
    )

    # set_target_properties(${NAME}_cpu PROPERTIES
    #     OUTPUT_NAME ${NAME}_cpu
    #     RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
    # )
endfunction(BUILD_EXEC)

foreach(test_op ${TEST_OPERATOR})
  BUILD_EXEC(${test_op})
endforeach()