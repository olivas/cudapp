//
// Created by olivas on 4/15/24.
//
#include <stdlib.h>
#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensornet.h>


#define HANDLE_ERROR(x)                                           \
{ const auto err = x;                                             \
  if( err != CUTENSORNET_STATUS_SUCCESS )                         \
  { printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("CUDA Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); \
    fflush(stdout);                                               \
  }                                                               \
};


struct GPUTimer
{
    GPUTimer(cudaStream_t stream): stream_(stream)
    {
        HANDLE_CUDA_ERROR(cudaEventCreate(&start_));
        HANDLE_CUDA_ERROR(cudaEventCreate(&stop_));
    }

    ~GPUTimer()
    {
        HANDLE_CUDA_ERROR(cudaEventDestroy(start_));
        HANDLE_CUDA_ERROR(cudaEventDestroy(stop_));
    }

    void start()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(start_, stream_));
    }

    float seconds()
    {
        HANDLE_CUDA_ERROR(cudaEventRecord(stop_, stream_));
        HANDLE_CUDA_ERROR(cudaEventSynchronize(stop_));
        float time;
        HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, start_, stop_));
        return time * 1e-3;
    }

private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
};


int main() {
    static_assert(sizeof(size_t) == sizeof(int64_t), "Please build this sample on a 64-bit architecture!");

    bool verbose = true;

    // Check cuTensorNet version
    const size_t cuTensornetVersion = cutensornetGetVersion();
    if (verbose)
        printf("cuTensorNet version: %ld\n", cuTensornetVersion);

    // Set GPU device
    int numDevices{0};
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    const int deviceId = 0;
    HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    if (verbose) {
        printf("===== device info ======\n");
        printf("GPU-name:%s\n", prop.name);
        printf("GPU-clock:%d\n", prop.clockRate);
        printf("GPU-memoryClock:%d\n", prop.memoryClockRate);
        printf("GPU-nSM:%d\n", prop.multiProcessorCount);
        printf("GPU-major:%d\n", prop.major);
        printf("GPU-minor:%d\n", prop.minor);
        printf("========================\n");
    }

    typedef float floatType;
    cudaDataType_t typeData = CUDA_R_32F;
    cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_32F;

    if (verbose)
        printf("Included headers and defined data types\n");
}