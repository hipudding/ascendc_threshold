#include <assert.h>
#include <sys/time.h>

#include <iostream>

#include "test_utils.h"
#include "threshold/threshold_opencv_kernel.h"

constexpr int threshold = 100;
constexpr int maxVal = 125;
constexpr int blockDim = 1;

#define TIMING(func)                                                         \
  struct timeval start, end;                                                 \
  gettimeofday(&start, NULL);                                                \
  {func} gettimeofday(&end, NULL);                                           \
  uint64_t time =                                                            \
      (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec); \
  std::cout << "operator execution time: " << time << "(µs)" << std::endl;

template<typename T>
void run_kernel(T* input, T* output, uint32_t size,
                ThresholdOpencvTilingData& tiling) {
#ifndef __CCE_KT_TEST__
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));
#endif

  tiling.maxVal = maxVal;
  tiling.thresh = threshold;
  tiling.totalLength = size;
  tiling.dtype = 1;

  uint8_t* inputDevice = upload(input, size * sizeof(T));
  uint8_t* outputDevice = upload(output, size * sizeof(T));
  uint8_t* tilingDevice = upload(&tiling, sizeof(ThresholdOpencvTilingData));

#ifdef __CCE_KT_TEST__
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(threshold_opencv, blockDim, tilingDevice, inputDevice, outputDevice);
#else
  TIMING(threshold_opencv_kernel(blockDim, nullptr, stream, tilingDevice, inputDevice,
                                 outputDevice);
         CHECK_ACL(aclrtSynchronizeStream(stream));)

#endif
  download(output, outputDevice, size * sizeof(T));
  ascendcFree(inputDevice);
  ascendcFree(outputDevice);
  ascendcFree(tilingDevice);

#ifndef __CCE_KT_TEST__
  CHECK_ACL(aclrtDestroyStream(stream));
#endif
}

template<typename T>
void run_thresh_trunc(T* input, T* output, uint32_t size) {
  std::cout << "run thresh trunc" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = 2;
  run_kernel(input, output, size, tiling);
}

template<typename T>
void check_result_thresh_trunc(T* input, T* output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      if(output[i] != threshold)
      assert(output[i] == threshold);
    } else {
      if(output[i] != input[i])
      assert(output[i] == input[i]);
    }
  }
  std::cout << "thresh trunc test passed" << std::endl;
}

template<typename T>
void run_thresh_binary(T* input, T* output, uint32_t size) {
  std::cout << "run thresh bianry" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = 0;
  run_kernel(input, output, size, tiling);
}

template<typename T>
void check_result_thresh_binary(T* input, T* output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == maxVal);
    } else {
      assert(output[i] == 0);
    }
  }
  std::cout << "thresh binary test passed" << std::endl;
}

template<typename T>
void run_thresh_binary_inv(T* input, T* output, uint32_t size) {
  std::cout << "run thresh bianry inv" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = 1;
  run_kernel(input, output, size, tiling);
}

template<typename T>
void check_result_thresh_binary_inv(T* input, T* output,
                                    uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == 0);
    } else {
      assert(output[i] == maxVal);
    }
  }
  std::cout << "thresh binary inv test passed" << std::endl;
}

template<typename T>
void run_thresh_tozero(T* input, T* output, uint32_t size) {
  std::cout << "run thresh tozero" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = 3;
  run_kernel(input, output, size, tiling);
}

template<typename T>
void check_result_thresh_tozero(T* input, T* output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == input[i]);
    } else {
      assert(output[i] == 0);
    }
  }
  std::cout << "thresh tozero test passed" << std::endl;
}

template<typename T>
void run_thresh_tozero_inv(T* input, T* output, uint32_t size) {
  std::cout << "run thresh tozero inv" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = 4;
  run_kernel(input, output, size, tiling);
}

template<typename T>
void check_result_thresh_tozero_inv(T* input, T* output,
                                    uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == 0);
    } else {
      assert(output[i] == input[i]);
    }
  }
  std::cout << "thresh tozero inv test passed" << std::endl;
}

int32_t main(int32_t argc, char* argv[]) {
#ifndef __CCE_KT_TEST__
  //CHECK_ACL(aclInit("/home/hua/code/ascendc_threshold/acl.json"));
  CHECK_ACL(aclInit(nullptr));
  aclrtContext context;
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
#endif

  uint32_t height = 4320;
  uint32_t width = 7680;
  uint32_t size = height * width;

  int8_t* input = (int8_t*)malloc(size * sizeof(int8_t));
  int8_t* output = (int8_t*)malloc(size * sizeof(int8_t));

  for (int i = 0; i < size; i++) {
    input[i] = (int8_t)(i%255);
  }

  run_thresh_binary(input, output, size);
  check_result_thresh_binary(input, output, size);

  run_thresh_binary_inv(input, output, size);
  check_result_thresh_binary_inv(input, output, size);

  run_thresh_trunc(input, output, size);
  check_result_thresh_trunc(input, output, size);

  run_thresh_tozero(input, output, size);
  check_result_thresh_tozero(input, output, size);

  run_thresh_tozero_inv(input, output, size);
  check_result_thresh_tozero_inv(input, output, size);

  free(input);
  free(output);
#ifndef __CCE_KT_TEST__
  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif
  return 0;
}
