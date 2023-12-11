INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB KERNEL_FILES
    threshold/*.cpp
)

add_subdirectory(cmake/cpu)
add_subdirectory(cmake/npu)
