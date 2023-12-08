set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE CCE)

add_library(ascendc_kernels SHARED ${KERNEL_FILES})

target_compile_options(ascendc_kernels PRIVATE
    -O2
    -std=c++17
)

target_compile_definitions(ascendc_kernels PRIVATE
    TILING_KEY_VAR=0
)

# set_target_properties(ascendc_kernels PROPERTIES
#     OUTPUT_NAME ascendc_kernels
#     LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
# )