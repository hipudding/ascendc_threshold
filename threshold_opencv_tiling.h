#ifndef THRESHOLD_OPENCV_TILING_H
#define THRESHOLD_OPENCV_TILING_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t AlignDiv32(int32_t n)
{
    return ((n + 31) & ~31) / 32;
}

inline __aicore__ int32_t Align32(int32_t n)
{
    return ((n + 31) & ~31);
}

inline __aicore__ int32_t DownAlign32(int32_t n)
{
    return (n & ~31);
}

struct ThresholdOpencvTilingData {
    int32_t maxVal;
    int32_t thres;
    uint32_t totalLength;
};

#define CHECK_ACL(x)                                                    \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#ifdef __CCE_KT_TEST__
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);
#else
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)                        \
    __ubuf__ uint8_t* tilingUbPointer = (__ubuf__ uint8_t*)get_imm(0);                          \
    copy_gm_to_ubuf(((__ubuf__ uint8_t*)(tilingUbPointer)), ((__gm__ uint8_t*)(tilingPointer)), \
        0, 1, AlignDiv32(sizeof(tilingStruct)), 0, 0);                                          \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingUbPointer);                      \
    pipe_barrier(PIPE_ALL);
#endif

// stub func, used to enable cpu mode in this code, will be deprecated soon
#define GET_TILING_DATA(tilingData, tilingPointer)                             \
    ThresholdOpencvTilingData tilingData;                                            \
    INIT_TILING_DATA(ThresholdOpencvTilingData, tilingDataPointer, tilingPointer);   \
    (tilingData).maxVal = tilingDataPointer->maxVal;                           \
    (tilingData).thres = tilingDataPointer->thres;                           \
    (tilingData).totalLength = tilingDataPointer->totalLength;
#endif // THRESHOLD_OPENCV_TILING_H