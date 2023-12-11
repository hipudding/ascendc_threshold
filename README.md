# Threshold operator written in AscendC

## Build threshold
This project default build with Ascend310p.
Add -DASCEND_INSTALL_PATH=/path/to/ascend/installed if cann installed in a custom dir.
```shell
mkdir build
cd build
cmake .. 
make
```

## Run threshol binary
All operators will compiled to two target, one is running on cpu simulator, another one is running on Ascend cores.
```shell
./kernel_test/test_threshold_cpu
./kernel_test/test_threshold_npu
```