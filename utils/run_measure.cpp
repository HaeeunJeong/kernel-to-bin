// run_kernel.cpp
// CUDA Driver API로 PTX 커널 실행 + 순수 커널 실행 시간 측정
// g++/nvcc로 빌드 가능 (nvcc 권장)
// nvcc -o run_kernel run_kernel.cpp -lcuda

#include <chrono>
#include <cuda.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    CUresult status = call;                                                    \
    if (status != CUDA_SUCCESS) {                                              \
      const char *errStr;                                                      \
      cuGetErrorName(status, &errStr);                                         \
      std::cerr << "CUDA error: " << errStr << " at line " << __LINE__         \
                << "\n";                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  // 1. CUDA 초기화
  CHECK_CUDA(cuInit(0));
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CUcontext context;
  CHECK_CUDA(cuCtxCreate(&context, 0, device));

  // 2. PTX 모듈 로드
  CUmodule module;
  CHECK_CUDA(cuModuleLoad(&module, "kernel.ptx"));

  // 3. 커널 핸들 가져오기 (PTX 내부 함수 이름에 맞게 변경 필요)
  CUfunction kernel;
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, "my_kernel"));

  // 4. (예시) 커널 인자 준비
  // 여기서는 단순히 int 값 하나 넘기는 경우
  int N = 1024;
  void *args[] = {&N};

  // 5. CUDA 이벤트 생성 (커널 실행 시간 측정용)
  CUevent start, stop;
  CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

  // 6. 커널 실행
  // gridDim=(1,1,1), blockDim=(N,1,1) 예시 → PTX에 맞게 조정 필요
  CHECK_CUDA(cuEventRecord(start, 0));
  CHECK_CUDA(cuLaunchKernel(kernel, 1, 1, 1, // gridDim
                            N, 1, 1,         // blockDim
                            0,               // sharedMemBytes
                            nullptr,         // stream
                            args,            // kernel arguments
                            nullptr));       // extra
  CHECK_CUDA(cuEventRecord(stop, 0));

  CHECK_CUDA(cuEventSynchronize(stop));

  // 7. 실행 시간 계산
  float elapsed_ms = 0.0f;
  CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
  std::cout << "Kernel execution time (ms): " << elapsed_ms << std::endl;

  // 8. 정리
  cuEventDestroy(start);
  cuEventDestroy(stop);
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return 0;
}
