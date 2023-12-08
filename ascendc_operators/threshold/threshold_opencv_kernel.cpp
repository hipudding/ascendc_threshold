#include "kernel_operator.h"
#include "threshold_opencv_kernel.h"

using namespace AscendC;

template <typename T>
class KernelThreshold {
 public:
  __aicore__ inline KernelThreshold(GM_ADDR x, GM_ADDR y, int32_t maxVal,
                                    int32_t thresh, uint32_t totalLength,
                                    ThresholdTypes threshType)
      : _maxVal(maxVal),
        _thresh(thresh),
        _totalLength(totalLength),
        _threshType(threshType),
        vecTiling(totalLength, GetBlockNum(), GetBlockIdx(), sizeof(T),
                  BUFFER_NUM * 2 + 1) {
    xGM.SetGlobalBuffer((__gm__ T*)x + vecTiling._blockOffset,
                        vecTiling._blockLength);
    yGM.SetGlobalBuffer((__gm__ T*)y + vecTiling._blockOffset,
                        vecTiling._blockLength);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, vecTiling._loopLength * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, vecTiling._loopLength * sizeof(T));
    pipe.InitBuffer(tmpQueue, 1, vecTiling._loopLength * sizeof(T));
  }

  __aicore__ inline void Process() {
    for (uint32_t loop = 0; loop < vecTiling._loopCount; loop++) {
      uint32_t offset = loop * vecTiling._loopLength;
      CopyIn(offset, vecTiling._loopLength);
      Compute(vecTiling._loopLength);
      CopyOut(offset, vecTiling._loopLength);
    }

    if (vecTiling._loopTailLength != 0) {
      uint32_t offset = vecTiling._loopCount * vecTiling._loopLength;
      CopyIn(offset, vecTiling._loopTailLength);
      Compute(vecTiling._loopTailLength);
      CopyOut(offset, vecTiling._loopTailLength);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGM[offset], len);
    inQueueX.EnQue<T>(xLocal);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopy(yGM[offset], yLocal, len);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void Compute(uint32_t len) {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    LocalTensor<uint8_t> mask = tmpQueue.AllocTensor<uint8_t>();
    Duplicate(yLocal, static_cast<T>(_thresh), len);
    switch (_threshType) {
      case THRESH_BINARY:
        Compare(mask, xLocal, yLocal, CMPMODE::LE, len);
        Duplicate(yLocal, static_cast<T>(0), len);
        Select(yLocal, mask, yLocal, static_cast<T>(_maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case THRESH_BINARY_INV:
        Compare(mask, xLocal, yLocal, CMPMODE::GT, len);
        Duplicate(yLocal, static_cast<T>(0), len);
        Select(yLocal, mask, yLocal, static_cast<T>(_maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case THRESH_TRUNC:
        Compare(mask, xLocal, yLocal, CMPMODE::LE, len);
        Select(yLocal, mask, xLocal, static_cast<T>(_maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case THRESH_TOZERO:
        Compare(mask, xLocal, yLocal, CMPMODE::GT, len);
        Select(yLocal, mask, xLocal, static_cast<T>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case THRESH_TOZERO_INV:
        Compare(mask, xLocal, yLocal, CMPMODE::LE, len);
        Select(yLocal, mask, xLocal, static_cast<T>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      default:
        break;
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
    tmpQueue.FreeTensor(mask);
  }

  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  TQue<QuePosition::VECIN, 1> tmpQueue;
  GlobalTensor<T> xGM, yGM;
  int32_t _maxVal;
  int32_t _thresh;
  uint32_t _totalLength;
  ThresholdTypes _threshType;
  VectorTiling vecTiling;
};

extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR x, GM_ADDR y,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling) {
  GET_TILING_DATA(ThresholdOpencvTilingData, tilingData, tiling);
  KernelThreshold<float> op(x, y, tilingData->maxVal, tilingData->thresh,
                            tilingData->totalLength, tilingData->threshType);
  op.Process();
}

#ifndef __CCE_KT_TEST__
void threshold_opencv_kernel(uint32_t blockDim, void* l2ctrl, void* stream,
                             uint8_t* x, uint8_t* y, uint8_t* workspace,
                             uint8_t* tiling) {
  threshold_opencv<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif