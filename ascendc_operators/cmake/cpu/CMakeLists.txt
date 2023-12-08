if (NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/cmake)
endif()

find_package(tikicpulib REQUIRED)

add_library(ascendc_kernels_debug SHARED
    ${KERNEL_FILES}
)

target_include_directories(ascendc_kernels_debug PRIVATE
    ${ASCEND_INSTALL_PATH}/acllib/include
)

target_link_libraries(ascendc_kernels_debug PRIVATE
    tikicpulib::${ASCEND_PRODUCT_TYPE}
    ascendcl
)

target_compile_options(ascendc_kernels_debug PRIVATE
    -O0
    -g
)

# set_target_properties(ascendc_kernels_debug PROPERTIES
#     OUTPUT_NAME ascendc_kernels_debug
#     LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
# )