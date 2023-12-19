#include <assert.h>
#include <sys/time.h>
#include <acl/acl.h>
#include <iostream>
#include <unistd.h>

#include "threshold_opencv_tiling.h"
#include "aclrtlaunch_threshold_opencv.h"

constexpr int threshold = 200;
constexpr int maxVal = 255;
constexpr int blockDim = 8;

#define TIMING(func)                                                         \
  struct timeval start, end;                                                 \
  gettimeofday(&start, NULL);                                                \
  {func} gettimeofday(&end, NULL);                                           \
  uint64_t time =                                                            \
      (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec); \
  std::cout << "operator execution time: " << time << "(µs)" << std::endl;

#define CHECK_ACL(x)                                                    \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

inline uint8_t* upload(void* src, size_t size) {
  uint8_t* dst;
  CHECK_ACL(aclrtMalloc((void**)&dst, size, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
  return dst;
}

inline void download(void* dst, void* src, size_t size) {
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
}

void run_kernel(float* input, float* output, uint32_t size,
                ThresholdOpencvTilingData& tiling) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  tiling.maxVal = maxVal;
  tiling.thresh = threshold;
  tiling.totalLength = size;
  uint8_t* inputDevice = upload(input, size * sizeof(float));
  uint8_t* outputDevice = upload(output, size * sizeof(float));
  uint8_t* tilingDevice = upload(&tiling, sizeof(ThresholdOpencvTilingData));

  TIMING(ACLRT_LAUNCH_KERNEL(threshold_opencv)(blockDim, stream, inputDevice,
                                               outputDevice, tilingDevice);
         CHECK_ACL(aclrtSynchronizeStream(stream));)

  download(output, outputDevice, size * sizeof(float));
  aclrtFree(inputDevice);
  aclrtFree(outputDevice);
  aclrtFree(tilingDevice);
}

void run_thresh_trunc(float* input, float* output, uint32_t size) {
  std::cout << "run thresh trunc" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = THRESH_TRUNC;
  run_kernel(input, output, size, tiling);
}

void check_result_thresh_trunc(float* input, float* output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == maxVal);
    } else {
      assert(output[i] == input[i]);
    }
  }
  std::cout << "thresh trunc test passed" << std::endl;
}

void run_thresh_binary(float* input, float* output, uint32_t size) {
  std::cout << "run thresh bianry" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = THRESH_BINARY;
  run_kernel(input, output, size, tiling);
}

void check_result_thresh_binary(float* input, float* output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == maxVal);
    } else {
      assert(output[i] == 0);
    }
  }
  std::cout << "thresh binary test passed" << std::endl;
}

void run_thresh_binary_inv(float* input, float* output, uint32_t size) {
  std::cout << "run thresh bianry inv" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = THRESH_BINARY_INV;
  run_kernel(input, output, size, tiling);
}

void check_result_thresh_binary_inv(float* input, float* output,
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

void run_thresh_tozero(float* input, float* output, uint32_t size) {
  std::cout << "run thresh tozero" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = THRESH_TOZERO;
  run_kernel(input, output, size, tiling);
}

void check_result_thresh_tozero(float* input, float* output, uint32_t size) {
  for (uint32_t i = 0; i < size; i++) {
    if (input[i] > threshold) {
      assert(output[i] == input[i]);
    } else {
      assert(output[i] == 0);
    }
  }
  std::cout << "thresh tozero test passed" << std::endl;
}

void run_thresh_tozero_inv(float* input, float* output, uint32_t size) {
  std::cout << "run thresh tozero inv" << std::endl;
  ThresholdOpencvTilingData tiling;
  tiling.threshType = THRESH_TOZERO_INV;
  run_kernel(input, output, size, tiling);
}

void check_result_thresh_tozero_inv(float* input, float* output,
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
  CHECK_ACL(aclInit(nullptr));
  aclrtContext context;
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));

  size_t tilingSize = sizeof(ThresholdOpencvTilingData);
  uint32_t height = 4320;
  uint32_t width = 7680;
  uint32_t size = height * width;
  float* input = (float*)malloc(size * sizeof(float));
  float* output = (float*)malloc(size * sizeof(float));

  for (int i = 0; i < size; i++) {
    input[i] = i;
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
  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
  return 0;
}