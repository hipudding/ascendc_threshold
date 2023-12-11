#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
#else
#include "tikicpulib.h"
#endif
#include <stdlib.h>

#define CHECK_ACL(x)                                                    \
  do {                                                                  \
    aclError __ret = x;                                                 \
    if (__ret != ACL_ERROR_NONE) {                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret \
                << std::endl;                                           \
    }                                                                   \
  } while (0);

inline uint8_t* upload(void* src, size_t size) {
  uint8_t* dst;
#ifdef __CCE_KT_TEST__
  dst = (uint8_t*)AscendC::GmAlloc(size);
  memcpy(dst, src, size);
#else
  CHECK_ACL(aclrtMalloc((void**)&dst, size, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
#endif
  return dst;
}

inline void download(void* dst, void* src, size_t size) {
#ifdef __CCE_KT_TEST__
  memcpy(dst, src, size);
#else
  CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
#endif
}

inline void ascendcFree(uint8_t* ptr) {
#ifdef __CCE_KT_TEST__
  AscendC::GmFree(ptr);
#else
  CHECK_ACL(aclrtFree(ptr));
#endif
}

#endif  // KERNEL_INTERFACE_H
