#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
extern void threshold_opencv_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y,
    uint8_t* workspace, uint8_t* tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
#endif
#include "iostream"
#include "stdlib.h"
#include "threshold_opencv_tiling.h"
#include <sys/time.h>

int32_t main(int32_t argc, char* argv[])
{
    size_t tilingSize = sizeof(ThresholdOpencvTilingData);
    uint32_t blockDim = 8;
    uint32_t height = 4320;
    uint32_t width = 7680;
    float* input = (float*)malloc(height * width * sizeof(float));
    float* output = (float*)malloc(height * width * sizeof(float));

    for(int i = 0;i<height * width;i++){
        input[i] = i;
    }
    
#ifdef __CCE_KT_TEST__
    ThresholdOpencvTilingData* tiling = (ThresholdOpencvTilingData*)AscendC::GmAlloc(tilingSize);
    tiling->maxVal = 255;
    tiling->thres = 255;
    tiling->totalLength = width * height;

    size_t inputByteSize = sizeof(float) * tiling->totalLength;
    size_t outputByteSize = sizeof(float) * tiling->totalLength;

    float* x = (float*)AscendC::GmAlloc(inputByteSize);
    float* y = (float*)AscendC::GmAlloc(outputByteSize);

    memcpy(x, input, inputByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(threshold_opencv, blockDim, (uint8_t*)x, (uint8_t*)y, nullptr, (uint8_t*)tiling); // use this macro for cpu debug

    memcpy(output, y, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    ThresholdOpencvTilingData* tilingHost;
    uint8_t *xDevice, *yDevice, *tilingDevice, *workspaceDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingSize));
    
    tilingHost->maxVal = 255;
    tilingHost->thres = 255;
    tilingHost->totalLength = width * height;

    size_t inputByteSize = sizeof(float) * tilingHost->totalLength;
    size_t outputByteSize = sizeof(float) * tilingHost->totalLength;

    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, input, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    struct timeval start, end;
    gettimeofday(&start, NULL);
    threshold_opencv_do(blockDim, nullptr, stream, xDevice, yDevice, workspaceDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    gettimeofday(&end, NULL);
    uint64_t time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    std::cout<<time<<std::endl;

    CHECK_ACL(aclrtMemcpy(output, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif

    for(int i = 0;i<height*width;i++) {
        if(input[i] <= 255) {
            if (output[i] != input[i]){
                std::cout<<"error: "<<i<<":"<<output[i]<<std::endl;
            }
        }
        else {
            if(output[i] != 255) {
                std::cout<<"error: "<<i<<":"<<output[i]<<std::endl;
            }
        }
    }
    std::cout<<"finish"<<std::endl;

    free(input);
    free(output);

    return 0;
}