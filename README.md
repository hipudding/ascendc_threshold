# Threshold operator written in AscendC

## Build threshold
```shell
mkdir build
cd build
cmake .. \
    -Dsmoke_testcase=threshold \
    -DASCEND_PRODUCT_TYPE="ascend310p" \
    -DASCEND_CORE_TYPE="AiCore" \
    -DASCEND_RUN_MODE="ONBOARD" \
    -DASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
cmake --build .. --target threshold_[cpu|npu]
```
