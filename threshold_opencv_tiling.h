#ifndef THRESHOLD_OPENCV_TILING_H
#define THRESHOLD_OPENCV_TILING_H

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

#endif  // THRESHOLD_OPENCV_TILING_H