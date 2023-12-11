#ifndef THRESHOLD_OPENCV_TILING_H
#define THRESHOLD_OPENCV_TILING_H

#ifdef __CCE_KT_TEST__
#include "kernel_utils.h"
#endif

enum ThresholdTypes {
  THRESH_BINARY,
  THRESH_BINARY_INV,
  THRESH_TRUNC,
  THRESH_TOZERO,
  THRESH_TOZERO_INV
};

struct ThresholdOpencvTilingData {
  int32_t maxVal;
  int32_t thresh;
  uint32_t totalLength;
  ThresholdTypes threshType;
};

#ifdef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR x, GM_ADDR y,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling);

#else
void threshold_opencv_kernel(uint32_t blockDim, void* l2ctrl, void* stream,
                             uint8_t* x, uint8_t* y, uint8_t* workspace,
                             uint8_t* tiling);
#endif
#endif  // THRESHOLD_OPENCV_TILING_H