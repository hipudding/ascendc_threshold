#ifndef THRESHOLD_OPENCV_TILING_H
#define THRESHOLD_OPENCV_TILING_H

#include "tiling_kernel.h"
struct ThresholdOpencvTilingData {
  int32_t maxVal;
  int32_t thres;
  uint32_t totalLength;
};

#endif //THRESHOLD_OPENCV_TILING_H