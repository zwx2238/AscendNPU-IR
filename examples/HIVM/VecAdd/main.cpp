/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file main.cpp
 * \brief Host main function to register and run hivm vadd example.
 */

#include "acl/acl.h"
#include "acl/error_codes/rt_error_codes.h"
#include "experiment/runtime/runtime/rt.h"

#include <cstdio>
#include <fstream>
#include <stdlib.h>

#define EXPECT_EQ(a, b, msg)                                                   \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      printf("[failed] %s\n", (msg));                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

char *readBinFile(const char *fileName, uint32_t *fileSize) {
  std::filebuf *pbuf;
  std::ifstream filestr;
  size_t size;
  filestr.open(fileName, std::ios::binary);
  if (!filestr) {
    printf("file:%s open failed!", fileName);
    return nullptr;
  }
  pbuf = filestr.rdbuf();
  size = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
  pbuf->pubseekpos(0, std::ios::in);
  char *buffer = (char *)malloc(size);
  if (!buffer) {
    printf("buffer malloc failed!");
    return nullptr;
  }
  pbuf->sgetn(buffer, size);
  *fileSize = size;

  printf("[success] file:%s read succ!\n", fileName);
  filestr.close();
  return buffer;
}

void *registerBinaryKernel(const char *filePath, char **buffer,
                           const char *stubFunc, const char *kernelName) {
  void *binHandle = nullptr;
  uint32_t bufferSize = 0;
  *buffer = readBinFile(filePath, &bufferSize);
  if (*buffer == nullptr) {
    printf("readBinFile: %s failed!\n", filePath);
    return nullptr;
  }

  rtDevBinary_t binary;
  binary.data = *buffer;
  binary.length = bufferSize;
  binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  binary.version = 0;
  rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtDevBinaryRegister: %s failed, errorCode=%d!\n", kernelName,
           rtRet);
    return nullptr;
  }
  rtRet = rtFunctionRegister(binHandle, (const void *)stubFunc, kernelName,
                             (void *)kernelName, 0);
  if (rtRet != RT_ERROR_NONE) {
    printf("rtFunctionRegister: %s failed, errorCode=%d!\n", kernelName, rtRet);
    return nullptr;
  }
  return binHandle;
}

int main() {
  // Initialize
  aclError error = ACL_RT_SUCCESS;
  error = aclInit(nullptr);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "init failed");

  error = aclrtSetDevice(0);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "set device failed");
  aclrtStream stream;
  error = aclrtCreateStream(&stream);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "create stream failed");
  printf("[success] initialize success\n");

  // Register the kernel
  char *buffer;
  // This is the function name in the .mlir
  const char *stubFunc = "add";
  // This is the path to the device binary
  void *binHandle =
      registerBinaryKernel("./kernel.o", &buffer, stubFunc, stubFunc);
  if (!binHandle)
    exit(1);
  printf("[success] register kernel success\n");

  // Prepare data
  int16_t expectedValue[] = {1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
  void *outputDevice = nullptr;
  error = aclrtMalloc((void **)&outputDevice, sizeof(expectedValue),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc output on device failed");

  int16_t input0Value[] = {0, 1, 2,  3,  4,  5,  6,  7,
                           8, 9, 10, 11, 12, 13, 14, 15};
  void *input0Device = nullptr;
  error = aclrtMalloc((void **)&input0Device, sizeof(input0Value),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc input0 on device failed");
  error = aclrtMemcpy((void *)input0Device, sizeof(input0Value), input0Value,
                      sizeof(input0Value), ACL_MEMCPY_HOST_TO_DEVICE);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcpy input0 on device failed");

  int16_t input1Value[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  void *input1Device = nullptr;
  error = aclrtMalloc((void **)&input1Device, sizeof(input1Value),
                      ACL_MEM_MALLOC_HUGE_FIRST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc input1 on device failed");
  error = aclrtMemcpy((void *)input1Device, sizeof(input1Value), input1Value,
                      sizeof(input1Value), ACL_MEMCPY_HOST_TO_DEVICE);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcpy input1 on device failed");
  printf("[success] memcpy host to device success\n");

  // Invoke the kernel
  void *args[] = {input0Device, input1Device, outputDevice};
  rtKernelLaunch(stubFunc, 1, static_cast<void *>(&args), sizeof(args), nullptr,
                 stream);
  error = aclrtSynchronizeStream(stream);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "stream synchronize failed");
  printf("[success] stream synchronize success\n");

  // Get the result
  int16_t *outHost = nullptr;
  error = aclrtMallocHost((void **)&outHost, sizeof(expectedValue));
  EXPECT_EQ(error, ACL_RT_SUCCESS, "alloc output on host failed");
  error = aclrtMemcpy(outHost, sizeof(expectedValue), outputDevice,
                      sizeof(expectedValue), ACL_MEMCPY_DEVICE_TO_HOST);
  EXPECT_EQ(error, ACL_RT_SUCCESS, "memcpy output to host failed");
  printf("[success] memcpy device to host success\n");

  for (int i = 0; i < sizeof(expectedValue) / sizeof(int16_t); i++) {
    printf("i%d\t Expect: %d\t\t\t\tResult: %d\n", i, expectedValue[i],
           outHost[i]);
  }
  printf("[success] compare output success\n");

  free(buffer);
  aclrtFreeHost(outHost);
  aclrtFree(outputDevice);
  aclrtFree(input0Device);
  aclrtFree(input1Device);

  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return 0;
}