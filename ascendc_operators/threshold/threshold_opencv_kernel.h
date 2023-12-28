#ifndef THRESHOLD_OPENCV_TILING_H
#define THRESHOLD_OPENCV_TILING_H

#ifdef __CCE_KT_TEST__
#include "kernel_utils.h"
#endif

#pragma pack(push, 8)
struct ThresholdOpencvTilingData {
  int32_t maxVal;
  int32_t thresh;
  uint32_t totalLength;
  uint8_t threshType;
  uint8_t dtype;
};
#pragma pack(pop)

#ifdef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR tiling, GM_ADDR x, GM_ADDR y);


#else
void threshold_opencv_kernel(uint32_t blockDim, void* l2ctrl, void* stream,
                             uint8_t* tiling, uint8_t* x, uint8_t* y);
#endif
#endif  // THRESHOLD_OPENCV_TILING_H
