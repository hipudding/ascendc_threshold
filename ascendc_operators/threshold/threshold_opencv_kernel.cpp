#include "threshold_opencv_kernel.h"

#include "kernel_operator.h"
#include "tiling_kernel.h"

using namespace AscendC;

/**
 * T: input data type.
 * C: data type for calculate.
 * if T != C, data should cast from T to C.
 */
template <typename T, typename C>
class KernelThreshold {
 public:
  __aicore__ inline KernelThreshold() {}
  __aicore__ inline void Init(GM_ADDR tilingGM, GM_ADDR x, GM_ADDR y) {
    auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
    auto tempTiling = (uint32_t*)&tilingData;
    for (int32_t i = 0;
         i < sizeof(ThresholdOpencvTilingData) / sizeof(uint32_t);
         ++i, ++tempTilingGM, ++tempTiling)
      *tempTiling = *tempTilingGM;

    uint64_t bytesPerElem = sizeof(C) * BUFFER_NUM * 2 + sizeof(uint8_t) * 1;

    // if T != C, two more cast buffer needed.
    if (!std::is_same<T, C>::value) bytesPerElem += sizeof(T) * 2;

    vecTiling.calculate(tilingData.totalLength, GetBlockNum(), GetBlockIdx(),
                        bytesPerElem, 256 / sizeof(C));

    xGM.SetGlobalBuffer((__gm__ T*)x + vecTiling._blockOffset,
                        vecTiling._blockLength);
    yGM.SetGlobalBuffer((__gm__ T*)y + vecTiling._blockOffset,
                        vecTiling._blockLength);

    // init cast buffer.
    if (!std::is_same<T, C>::value) {
      pipe.InitBuffer(castBufferX, 1, vecTiling._loopLength * sizeof(T));
      pipe.InitBuffer(castBufferY, 1, vecTiling._loopLength * sizeof(T));
    }

    pipe.InitBuffer(inQueueX, BUFFER_NUM, vecTiling._loopLength * sizeof(C));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, vecTiling._loopLength * sizeof(C));
    pipe.InitBuffer(maskBuffer, 1, vecTiling._loopLength * sizeof(uint8_t));
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
    if (std::is_same<T, C>::value) {
      LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
      DataCopy(xLocal, xGM[offset], len);
      inQueueX.EnQue(xLocal);
    } else {
      LocalTensor<C> xLocal = inQueueX.AllocTensor<C>();
      LocalTensor<T> xCast = castBufferX.AllocTensor<T>();
      DataCopy(xCast, xGM[offset], len);
      castBufferX.EnQue(xCast);
      xCast = castBufferX.DeQue<T>();
      Cast(xLocal, xCast, RoundMode::CAST_NONE, len);
      castBufferX.FreeTensor(xCast);
      inQueueX.EnQue(xLocal);
    }
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
    if (std::is_same<T, C>::value) {
      LocalTensor<T> yLocal = outQueueY.DeQue<T>();
      DataCopy(yGM[offset], yLocal, len);
      outQueueY.FreeTensor(yLocal);
    } else {
      LocalTensor<C> yLocal = outQueueY.DeQue<C>();
      LocalTensor<T> yCast = castBufferY.AllocTensor<T>();
      Cast(yCast, yLocal, RoundMode::CAST_ROUND, len);
      castBufferY.EnQue(yCast);
      yCast = castBufferY.DeQue<T>();
      DataCopy(yGM[offset], yCast, len);
      castBufferY.FreeTensor(yCast);
      outQueueY.FreeTensor(yLocal);
    }
  }

  __aicore__ inline void CompareWrap(const LocalTensor<uint8_t>& dstLocal,
                                     const LocalTensor<C>& src0Local,
                                     const LocalTensor<C>& src1Local,
                                     CMPMODE cmpMode, uint32_t calCount) {
    uint32_t batchCount = 256 / sizeof(C);
    uint32_t tailCount = calCount % batchCount;

    Compare(dstLocal, src0Local, src1Local, cmpMode, calCount - tailCount);
    if (tailCount != 0) {
      BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
      uint32_t tailIdx = calCount - tailCount;
      uint32_t maskIdx = tailIdx / sizeof(uint8_t);
      Compare(dstLocal[maskIdx], src0Local[tailIdx], src1Local[tailIdx], cmpMode, tailCount, 1,
              repeatParams);
    }
  }

  __aicore__ inline void SelectWrap(const LocalTensor<C>& dstLocal,
                                    const LocalTensor<uint8_t>& selMask,
                                    const LocalTensor<C>& src0Local,
                                    C src1Local, SELMODE selMode,
                                    uint32_t calCount) {
    uint32_t batchCount = 256 / sizeof(C);
    uint32_t tailCount = calCount % batchCount;

    Select(dstLocal, selMask, src0Local, src1Local, selMode,
           calCount - tailCount);
    if (tailCount != 0) {
      BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
      uint32_t tailIdx = calCount - tailCount;
      uint32_t maskIdx = tailIdx / sizeof(uint8_t);
      Select(dstLocal[tailIdx], selMask[maskIdx], src0Local[tailIdx], src1Local, selMode, tailCount,
             1, repeatParams);
    }
  }

  __aicore__ inline void Compute(uint32_t len) {
    LocalTensor<C> xLocal = inQueueX.DeQue<C>();
    LocalTensor<C> yLocal = outQueueY.AllocTensor<C>();
    LocalTensor<uint8_t> mask = maskBuffer.AllocTensor<uint8_t>();
    Duplicate(yLocal, static_cast<C>(tilingData.thresh), len);
    switch (tilingData.threshType) {
      case 0:
        CompareWrap(mask, xLocal, yLocal, CMPMODE::LE, len);
        Duplicate(yLocal, static_cast<C>(0), len);
        SelectWrap(yLocal, mask, yLocal, static_cast<C>(tilingData.maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case 1:
        CompareWrap(mask, xLocal, yLocal, CMPMODE::GT, len);
        Duplicate(yLocal, static_cast<C>(0), len);
        SelectWrap(yLocal, mask, yLocal, static_cast<C>(tilingData.maxVal),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case 2:
        CompareWrap(mask, xLocal, yLocal, CMPMODE::LE, len);
        SelectWrap(yLocal, mask, xLocal, static_cast<C>(tilingData.thresh),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case 3:
        CompareWrap(mask, xLocal, yLocal, CMPMODE::GT, len);
        SelectWrap(yLocal, mask, xLocal, static_cast<C>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      case 4:
        CompareWrap(mask, xLocal, yLocal, CMPMODE::LE, len);
        SelectWrap(yLocal, mask, xLocal, static_cast<C>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
        break;
      default:
        break;
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
    maskBuffer.FreeTensor(mask);
  }

  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  TQue<QuePosition::VECIN, 1> castBufferX, maskBuffer, castBufferY;

  GlobalTensor<T> xGM, yGM;
  VectorTiling vecTiling;
  ThresholdOpencvTilingData tilingData;
};

extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR tiling,
                                                       GM_ADDR x, GM_ADDR y) {
  KernelThreshold<uint8_t, half> op;
  op.Init(tiling, x, y);
  op.Run();
  dcci(tiling, 1);
}

#ifndef __CCE_KT_TEST__
void threshold_opencv_kernel(uint32_t blockDim, void* l2ctrl, void* stream,
                             uint8_t* tiling, uint8_t* x, uint8_t* y) {
  threshold_opencv<<<blockDim, l2ctrl, stream>>>(tiling, x, y);
}
#endif
