#ifndef TILING_KERNEL_H
#define TILING_KERNEL_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t AlignDiv32(int32_t n) {
  return ((n + 31) & ~31) / 32;
}

inline __aicore__ int32_t Align32Ceil(int32_t n) { return ((n + 31) & ~31); }

inline __aicore__ int32_t Align32Floor(int32_t n) { return (n & ~31); }

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUF_LEN = 248 * 1024;

struct VectorTiling {
  __aicore__ inline VectorTiling(uint64_t totalLength, uint64_t blockNum,
                                 uint64_t blockIdx, uint32_t variableSize,
                                 uint32_t variableSetCount)
      : _totalLength(totalLength),
        _blockNum(blockNum),
        _blockIdx(blockIdx),
        _variableSize(variableSize),
        _variableSetCount(variableSetCount) {
    GetBlockLengthAndOffset();
    GetLoopLengthAndCount();
  }

  __aicore__ inline void GetBlockLengthAndOffset() {
    _blockLength = _totalLength / _blockNum;
    uint64_t tailLength = _totalLength % _blockNum;

    // Distribute the tail data
    if (_blockIdx < tailLength) {
      _blockLength++;
    }

    _blockOffset = _blockLength * _blockIdx;

    /**
     * @brief If block index is after blocks which get taile data, should add
     * tailLength. Eg: BlockNum = 5, totalLength = 32;  32 / 5 = 6, 32 % 5 = 2;
     * Block0~1: blockLength = 6 + 1 = 7; block0's offset = 0 * 7 = 0;
     *           block1's offset = 1 * 7 = 7;
     * Block2~4: blockLength = 6; block2's offset = 2 * 6 + 2 = 14;
     *           block3's offset = 3 * 6 + 2 = 20;
     */
    if (_blockIdx >= tailLength) {
      _blockOffset += tailLength;
    }
  }

  __aicore__ inline void GetLoopLengthAndCount() {
    _loopLength = Align32Floor(UB_BUF_LEN / _variableSize / _variableSetCount);
    _loopCount = _blockLength / _loopLength;
    _tailLength = _blockLength - (_loopLength * _loopCount);
  }

  uint64_t _totalLength;
  uint64_t _blockNum;
  uint64_t _blockIdx;
  uint32_t _variableSize;
  uint32_t _variableSetCount;
  uint32_t _blockLength;
  uint32_t _blockOffset;
  uint32_t _loopLength;
  uint32_t _loopCount;
  uint32_t _tailLength;
};

#define CHECK_ACL(x)                                                    \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

#define CONVERT_TILING_DATA(tilingStruct, tilingData, tilingPointer) \
  __ubuf__ tilingStruct* tilingData =                                \
      reinterpret_cast<__ubuf__ tilingStruct*>(                      \
          (__ubuf__ uint8_t*)(tilingPointer));

#ifdef __CCE_KT_TEST__
#define INIT_TILING_DATA(tilingStruct, tilingData, tilingPointer) \
  CONVERT_TILING_DATA(tilingStruct, tilingData, tilingPointer);
#else
#define GET_TILING_DATA(tilingStruct, tilingData, tilingPointer)    \
  __ubuf__ uint8_t* tilingUbPointer = (__ubuf__ uint8_t*)get_imm(0); \
  copy_gm_to_ubuf(((__ubuf__ uint8_t*)(tilingUbPointer)),            \
                  ((__gm__ uint8_t*)(tilingPointer)), 0, 1,          \
                  AlignDiv32(sizeof(tilingStruct)), 0, 0);           \
  CONVERT_TILING_DATA(tilingStruct, tilingData, tilingUbPointer);    \
  pipe_barrier(PIPE_ALL);
#endif

#endif  // TILING_KERNEL_H