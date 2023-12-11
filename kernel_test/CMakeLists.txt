INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../ascendc_operators)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../ascendc_operators/include)
INCLUDE_DIRECTORIES(${ASCEND_INSTALL_PATH}/include)

set(lib_ascendcl "${ASCEND_INSTALL_PATH}/acllib/lib64")
find_library(found_lib_ascendcl NAMES ascendcl PATHS ${lib_ascendcl} NO_DEFAULT_PATH)
if(found_lib_ascendcl)
    set(lib_ascendcl ${found_lib_ascendcl})
    message(STATUS "libascendcl.so is found at ${lib_ascendcl}")
else()
    message(FATAL_ERROR "Missing libascendcl.so.")
endif()

function(BUILD_TARGET OP_NAME ARCH LIBS)
    add_executable(test_${OP_NAME}_${ARCH}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${OP_NAME}.cpp
    )
    target_compile_options(test_${OP_NAME}_${ARCH} PRIVATE
        -O2
        -std=c++17
    )
    target_link_libraries(test_${OP_NAME}_${ARCH} PRIVATE ${LIBS})
    target_compile_definitions(test_${OP_NAME}_${ARCH} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endfunction(BUILD_TARGET)

set(CPU_LINK_LIBS         
    tikicpulib::${ASCEND_PRODUCT_TYPE}
    ${lib_ascendcl}
    ascendc_kernels_debug
    )

set(NPU_LINK_LIBS         
    ${lib_ascendcl}
    ascendc_kernels
    )

set(TEST_OPERATOR 
    threshold
    )

foreach(test_op ${TEST_OPERATOR})
    BUILD_TARGET(${test_op} cpu "${CPU_LINK_LIBS}")
    BUILD_TARGET(${test_op} npu "${NPU_LINK_LIBS}")
endforeach()
