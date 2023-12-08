cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)

set(ASCEND_PRODUCT_TYPE ascend310p)
set(ASCEND_CORE_TYPE AiCore)
set(ASCEND_RUN_MODE ONBOARD)
set(ASCEND_INSTALL_PATH /usr/local/Ascend/ascend-toolkit/latest)

set(CCE_CMAKE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${CCE_CMAKE_PATH})
project(ascendc_operators LANGUAGES CCE CXX)

add_subdirectory(ascendc_operators)
add_subdirectory(kernel_test)