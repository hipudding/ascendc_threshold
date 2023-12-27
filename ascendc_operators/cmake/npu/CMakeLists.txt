set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE CCE)

add_library(ascendc_kernels SHARED ${KERNEL_FILES})

target_compile_options(ascendc_kernels PRIVATE
    -O2
    -std=c++17
)