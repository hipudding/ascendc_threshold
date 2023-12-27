#ifndef TILING_KERNEL_H
#define TILING_KERNEL_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif



inline __aicore__ int32_t AlignNCeil(int32_t n, int32_t align) { return ((n + align) & ~(align-1)); }

inline __aicore__ int32_t AlignNFloor(int32_t n, int32_t align) { return (n & ~(align-1)); }

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUF_LEN = 248 * 1024;

struct VectorTiling {
  __aicore__ inline void calculate(uint64_t totalLength, uint64_t blockNum,
                                   uint64_t blockIdx, uint64_t variableBytesPerElem, uint32_t align) {
    _totalLength = totalLength, _blockNum = blockNum;
    _blockIdx = blockIdx;
    _variableBytesPerElem = variableBytesPerElem;
    _blockLength = 0;
    _blockOffset = 0;
    _align = align;
    GetBlockLengthAndOffset();
    GetLoopLengthAndCount();
#ifdef __CCE_KT_TEST__
    std::cout << "Block(" << _blockIdx << "): BlockLength = " << _blockLength
              << ", BlockOffset = " << _blockOffset
              << ", LoopLength = " << _loopLength
              << ", LoopCount = " << _loopCount
              << ", LoopTailLength = " << _loopTailLength << std::endl;
#endif
  }

  __aicore__ inline void GetBlockLengthAndOffset() {
    // Data should Align by 32B.
    uint32_t fullBlockLength = AlignNCeil(_totalLength / _blockNum, 32);
    // Some core may get no data after Align32 Ceil.
    uint32_t fullBlockNum = _totalLength / fullBlockLength;
    uint32_t blockTailLength = _totalLength % fullBlockLength;

    if (_blockIdx < fullBlockNum) {
      _blockLength = fullBlockLength;
      _blockOffset = _blockIdx * _blockLength;
      // Last block must less than full block num.
    } else if (blockTailLength != 0 && _blockIdx == fullBlockNum) {
      _blockLength = blockTailLength;
      _blockOffset = _blockIdx * fullBlockLength;
    }
  }

  /**
   * @brief Get length for one loop and loop count.
   * Use as much UB buf as possible.
   */
  __aicore__ inline void GetLoopLengthAndCount() {
    _loopLength = AlignNFloor(UB_BUF_LEN / _variableBytesPerElem, _align);
    _loopCount = _blockLength / _loopLength;
    _loopTailLength = _blockLength - (_loopLength * _loopCount);
  }

  uint64_t _totalLength;
  uint64_t _blockNum;
  uint64_t _blockIdx;
  uint64_t _variableBytesPerElem;
  uint32_t _blockLength;
  uint32_t _blockOffset;
  uint32_t _loopLength;
  uint32_t _loopCount;
  uint32_t _loopTailLength;
  uint32_t _align;
};

#endif  // TILING_KERNEL_H
