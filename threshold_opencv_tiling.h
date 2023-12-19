#ifndef THRESHOLD_OPENCV_TILING_H
#define THRESHOLD_OPENCV_TILING_H

enum ThresholdTypes {
  THRESH_BINARY,
  THRESH_BINARY_INV,
  THRESH_TRUNC,
  THRESH_TOZERO,
  THRESH_TOZERO_INV
};

#pragma pack(push, 8)
struct ThresholdOpencvTilingData {
  int32_t maxVal;
  int32_t thresh;
  uint32_t totalLength;
  ThresholdTypes threshType;
};
#pragma pack(pop)
#endif  // THRESHOLD_OPENCV_TILING_H