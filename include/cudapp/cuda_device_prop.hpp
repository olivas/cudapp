//
// Created by olivas on 4/15/24.
//

#ifndef CUDAPP_CUDA_DEVICE_PROP_HPP
#define CUDAPP_CUDA_DEVICE_PROP_HPP

#include <array>
#include <string>

struct cuda_device_prop {
    std::string name;
    std::string uuid; // 16 byte unique identifier
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    unsigned regsPerBlock;
    unsigned warpSize;
    size_t memPitch;
    unsigned maxThreadsPerBlock;
    std::array<unsigned, 3> maxThreadsDim;
    std::array<unsigned, 3> maxGridSize;
    unsigned clockRate;
    size_t totalConstMem;
    unsigned major;
    unsigned minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    unsigned deviceOverlap;
    unsigned multiProcessorCount;
    unsigned kernelExecTimeoutEnabled;
    unsigned integrated;
    unsigned canMapHostMemory;
    unsigned computeMode;
    unsigned maxTexture1D;
    unsigned maxTexture1DMipmap;
    unsigned maxTexture1DLinear;
    std::array<unsigned, 2> maxTexture2D;
    std::array<unsigned, 2> maxTexture2DMipmap;
    std::array<unsigned, 3> maxTexture2DLinear;
    std::array<unsigned, 2> maxTexture2DGather;
    std::array<unsigned, 3> maxTexture3D;
    std::array<unsigned, 3> maxTexture3DAlt;
    unsigned maxTextureCubemap;
    std::array<unsigned, 2> maxTexture1DLayered;
    std::array<unsigned, 3> maxTexture2DLayered;
    std::array<unsigned, 2> maxTextureCubemapLayered;
    unsigned maxSurface1D;
    std::array<unsigned, 2> maxSurface2D;
    std::array<unsigned, 3> maxSurface3D;
    std::array<unsigned, 2> maxSurface1DLayered;
    std::array<unsigned, 3> maxSurface2DLayered;
    unsigned maxSurfaceCubemap;
    std::array<unsigned, 2> maxSurfaceCubemapLayered;
    size_t surfaceAlignment;
    unsigned concurrentKernels;
    unsigned ECCEnabled;
    unsigned pciBusID;
    unsigned pciDeviceID;
    unsigned pciDomainID;
    unsigned tccDriver;
    unsigned asyncEngineCount;
    unsigned unifiedAddressing;
    unsigned memoryClockRate;
    unsigned memoryBusWidth;
    unsigned l2CacheSize;
    unsigned persistingL2CacheMaxSize;
    unsigned maxThreadsPerMultiProcessor;
    unsigned streamPrioritiesSupported;
    unsigned globalL1CacheSupported;
    unsigned localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    unsigned regsPerMultiprocessor;
    unsigned managedMemory;
    unsigned isMultiGpuBoard;
    unsigned multiGpuBoardGroupID;
    unsigned singleToDoublePrecisionPerfRatio;
    unsigned pageableMemoryAccess;
    unsigned concurrentManagedAccess;
    unsigned computePreemptionSupported;
    unsigned canUseHostPointerForRegisteredMem;
    unsigned cooperativeLaunch;
    unsigned cooperativeMultiDeviceLaunch;
    unsigned pageableMemoryAccessUsesHostPageTables;
    unsigned directManagedMemAccessFromHost;
    unsigned accessPolicyMaxWindowSize;
};


#endif //CUDAPP_CUDA_DEVICE_PROP_HPP
