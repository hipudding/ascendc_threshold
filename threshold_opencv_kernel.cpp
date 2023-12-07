#include "kernel_operator.h"
#include "threshold_opencv_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUF_LEN = 248 * 1024;

__aicore__ inline void GetBlockLength(uint32_t blockNum, uint32_t blockIdx,
                           uint32_t totalLength, uint32_t& blockLength,
                           uint32_t& offset) {
  blockLength = totalLength / blockNum;
  uint32_t tail = totalLength % blockNum;

  if (blockIdx < tail) {
    blockLength++;
  }

  offset = blockLength * blockIdx;

  if (blockIdx >= tail) {
    offset += tail;
  }
}

__aicore__ inline void GetDataLengthPerLoop(uint32_t singleVarSetSize,
                                 uint32_t sizeofT,
                                 uint32_t buffer_num, uint32_t blockLength,
                                 uint32_t& dataLengthPerLoop,
                                 uint32_t& loopCount, uint32_t& tailLength) {
  dataLengthPerLoop = DownAlign32(UB_BUF_LEN / sizeofT / singleVarSetSize / buffer_num);
  loopCount = blockLength / dataLengthPerLoop;
  tailLength = blockLength - (dataLengthPerLoop * loopCount);
}

class KernelThreshold {
 public:
  __aicore__ inline KernelThreshold() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                              int32_t maxVal, int32_t thres, uint32_t totalLength) {
    this->maxVal = maxVal;
    this->thres = thres;
    uint64_t blockNum = GetBlockNum();
    uint64_t blockIdx = GetBlockIdx();
    ASSERT(blockNum != 0 && "block dim can not be zero");
    uint32_t offset;
    GetBlockLength(blockNum, blockIdx, totalLength, this->blockLength, offset);

    xGM.SetGlobalBuffer((__gm__ float*)x + offset, this->blockLength);
    yGM.SetGlobalBuffer((__gm__ float*)y + offset, this->blockLength);

    GetDataLengthPerLoop(3, sizeof(float), BUFFER_NUM, this->blockLength,
                         this->dataLengthPerLoop, this->loopCount,
                         this->tailLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, Align32(this->dataLengthPerLoop * sizeof(float)));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, Align32(this->dataLengthPerLoop * sizeof(float)));
    pipe.InitBuffer(tmpQueue, 1, Align32(this->dataLengthPerLoop * sizeof(uint8_t)));
  }

  __aicore__ inline void Process() {
    for (uint32_t loop = 0; loop < this->loopCount; loop++) {
        uint32_t offset = loop * this->dataLengthPerLoop;
        CopyIn(offset, this->dataLengthPerLoop);
        Compute(this->dataLengthPerLoop);
        CopyOut(offset, this->dataLengthPerLoop);
    }

    if(this->tailLength != 0) {
        uint32_t offset = this->loopCount * this->dataLengthPerLoop;
        CopyIn(offset, this->tailLength);
        Compute(this->tailLength);
        CopyOut(offset, this->tailLength);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t offset, uint32_t len) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    DataCopy(xLocal, xGM[offset],len);
    inQueueX.EnQue<float>(xLocal);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t len) {
    LocalTensor<float> yLocal = outQueueY.DeQue<float>();
    DataCopy(yGM[offset], yLocal, len);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void Compute(uint32_t len) {
    LocalTensor<float> xLocal = inQueueX.DeQue<float>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    LocalTensor<uint8_t> mask = tmpQueue.AllocTensor<uint8_t>();

    Duplicate(yLocal, (float)this->thres, len);
    Compare(mask, xLocal, yLocal, CMPMODE::LE, len);

    Select(yLocal, mask, xLocal, (float)this->maxVal, SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    
    //Add(yLocal, xLocal, xLocal, len);
    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
    tmpQueue.FreeTensor(mask);
  }

  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
  TQue<QuePosition::VECIN, 1> tmpQueue;
  GlobalTensor<float> xGM, yGM;
  int32_t maxVal;
  int32_t thres;
  uint32_t totalLength;
  uint32_t blockLength;
  uint32_t dataLengthPerLoop;
  uint32_t tailLength;
  uint32_t loopCount;
};

extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR x, GM_ADDR y,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  KernelThreshold op;
  op.Init(x, y, tilingData.maxVal, tilingData.thres, tilingData.totalLength);
  op.Process();
}

#ifndef __CCE_KT_TEST__
void threshold_opencv_do(uint32_t blockDim, void* l2ctrl, void* stream,
                         uint8_t* x, uint8_t* y, uint8_t* workspace,
                         uint8_t* tiling) {
  threshold_opencv<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif