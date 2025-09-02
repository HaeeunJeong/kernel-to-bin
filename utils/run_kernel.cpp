#include <cuda.h>
#include <iostream>

int main() {
  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;

  // CUDA 초기화
  cuInit(0);
  cuDeviceGet(&device, 0); // 첫 번째 GPU
  cuCtxCreate(&context, 0, device);

  // PTX 모듈 로드
  cuModuleLoad(&module, "kernel.ptx");

  // 커널 함수 핸들 가져오기
  cuModuleGetFunction(&kernel, module, "my_kernel");

  // 실행 인자 준비
  void *args[] = {/* 커널 파라미터 포인터들 */};

  // 커널 실행 (1D grid/block 예시)
  cuLaunchKernel(kernel, 1, 1, 1, // gridDim
                 32, 1, 1,        // blockDim
                 0, nullptr, args, nullptr);

  cuCtxSynchronize();

  std::cout << "Kernel executed!\n";
  return 0;
}
