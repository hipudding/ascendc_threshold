#include "kernel_operator.h"
#include "threshold_opencv_tiling.h"
#include "tiling_kernel.h"

using namespace AscendC;

template <typename T>
class KernelThreshold {
 public:
  __aicore__ inline KernelThreshold() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM) {
    auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
    auto tempTiling = (uint32_t*)&tilingData;
    for (int32_t i = 0;
         i < sizeof(ThresholdOpencvTilingData) / sizeof(uint32_t);
         ++i, ++tempTilingGM, ++tempTiling) {
      *tempTiling = *tempTilingGM;
    }

    vecTiling.calculate(tilingData.totalLength, GetBlockNum(), GetBlockIdx(),
                        sizeof(T), BUFFER_NUM * 2 + 1);

    xGM.SetGlobalBuffer((__gm__ T*)x + vecTiling._blockOffset,
                        vecTiling._blockLength);
    yGM.SetGlobalBuffer((__gm__ T*)y + vecTiling._blockOffset,
                        vecTiling._blockLength);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, vecTiling._loopLength * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, vecTiling._loopLength * sizeof(T));
    pipe.InitBuffer(tmpQueue, 1, vecTiling._loopLength * sizeof(T));
  }

  __aicore__ inline void Run() {
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
    Duplicate(yLocal, static_cast<T>(tilingData.thresh), len);
    switch (tilingData.threshType) {
      case THRESH_BINARY:
        Compare(mask, xLocal, yLocal, CMPMODE::LE, len);
        Duplicate(yLocal, static_cast<T>(0), len);
        Select(yLocal, mask, yLocal, static_cast<T>(tilingData.maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case THRESH_BINARY_INV:
        Compare(mask, xLocal, yLocal, CMPMODE::GT, len);
        Duplicate(yLocal, static_cast<T>(0), len);
        Select(yLocal, mask, yLocal, static_cast<T>(tilingData.maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case THRESH_TRUNC:
        Compare(mask, xLocal, yLocal, CMPMODE::LE, len);
        Select(yLocal, mask, xLocal, static_cast<T>(tilingData.maxVal),
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
  VectorTiling vecTiling;
  ThresholdOpencvTilingData tilingData;
};

extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR x, GM_ADDR y,
                                                       GM_ADDR tiling) {
  KernelThreshold<float> op;
  op.Init(x, y, tiling);
  op.Run();
  dcci(tiling, 1);
}