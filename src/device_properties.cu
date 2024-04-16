#include <string>
#include <array>
#include <iostream>

#include <cudapp/device_properties.hpp>
#include <cudapp/check_error.cuh>

#include <cuda.h>
#include <cuda_runtime_api.h>

using std::optional;
using std::string;
using std::array;
using std::cout;
using std::endl;

cudapp::DeviceProperties::DeviceProperties(unsigned device_number) :
        device_number_{device_number} {
    // check that it's a valid number
    set_device_properties(device_number_);
}

void
cudapp::DeviceProperties::change_device_number(unsigned int device_number) {
    device_number_ = device_number;
    set_device_properties(device_number_);
}

void
cudapp::DeviceProperties::set_device_properties(unsigned int device_number) {
    cudaDeviceProp cudevprops;
    CHECK_ERROR(cudaGetDeviceProperties(&cudevprops, device_number));

    // now cast and translate
    device_properties.name = string(cudevprops.name);
    //device_properties.uuid; // 16 byte unique identifier
    device_properties.totalGlobalMem = static_cast<unsigned>(cudevprops.totalGlobalMem);
    device_properties.sharedMemPerBlock = static_cast<unsigned>(cudevprops.sharedMemPerBlock);
    device_properties.regsPerBlock = static_cast<unsigned>(cudevprops.regsPerBlock);
    device_properties.warpSize = static_cast<unsigned>(cudevprops.warpSize);
    device_properties.memPitch = static_cast<unsigned>(cudevprops.memPitch);
    device_properties.maxThreadsPerBlock = static_cast<unsigned>(cudevprops.maxThreadsPerBlock);
    device_properties.maxThreadsDim[0] = static_cast<unsigned>(cudevprops.maxThreadsDim[0]);
    device_properties.maxThreadsDim[1] = static_cast<unsigned>(cudevprops.maxThreadsDim[1]);
    device_properties.maxGridSize[0] = static_cast<unsigned>(cudevprops.maxGridSize[0]);
    device_properties.maxGridSize[1] = static_cast<unsigned>(cudevprops.maxGridSize[1]);
    device_properties.clockRate = static_cast<unsigned>(cudevprops.clockRate);
    device_properties.totalConstMem = static_cast<unsigned>(cudevprops.totalConstMem);
    device_properties.major = static_cast<unsigned>(cudevprops.major);
    device_properties.minor = static_cast<unsigned>(cudevprops.minor);
    device_properties.textureAlignment = static_cast<unsigned>(cudevprops.textureAlignment);
    device_properties.texturePitchAlignment = static_cast<unsigned>(cudevprops.texturePitchAlignment);
    device_properties.deviceOverlap = static_cast<unsigned>(cudevprops.deviceOverlap);
    device_properties.multiProcessorCount = static_cast<unsigned>(cudevprops.multiProcessorCount);
    device_properties.kernelExecTimeoutEnabled = static_cast<unsigned>(cudevprops.kernelExecTimeoutEnabled);
    device_properties.integrated = static_cast<unsigned>(cudevprops.integrated);
    device_properties.canMapHostMemory = static_cast<unsigned>(cudevprops.canMapHostMemory);
    device_properties.computeMode = static_cast<unsigned>(cudevprops.computeMode);
    device_properties.maxTexture1D = static_cast<unsigned>(cudevprops.maxTexture1D);
    device_properties.maxTexture1DMipmap = static_cast<unsigned>(cudevprops.maxTexture1DMipmap);
    device_properties.maxTexture1DLinear = static_cast<unsigned>(cudevprops.maxTexture1DLinear);
    device_properties.maxTexture2D[0] = static_cast<unsigned>(cudevprops.maxTexture2D[0]);
    device_properties.maxTexture2D[1] = static_cast<unsigned>(cudevprops.maxTexture2D[1]);
    device_properties.maxTexture2DMipmap[0] = static_cast<unsigned>(cudevprops.maxTexture2DMipmap[0]);
    device_properties.maxTexture2DMipmap[1] = static_cast<unsigned>(cudevprops.maxTexture2DMipmap[1]);
    device_properties.maxTexture2DLinear[0] = static_cast<unsigned>(cudevprops.maxTexture2DLinear[0]);
    device_properties.maxTexture2DLinear[1] = static_cast<unsigned>(cudevprops.maxTexture2DLinear[1]);
    device_properties.maxTexture2DLinear[2] = static_cast<unsigned>(cudevprops.maxTexture2DLinear[2]);
    device_properties.maxTexture2DGather[0] = static_cast<unsigned>(cudevprops.maxTexture2DGather[0]);
    device_properties.maxTexture2DGather[1] = static_cast<unsigned>(cudevprops.maxTexture2DGather[1]);
    device_properties.maxTexture3D[0] = static_cast<unsigned>(cudevprops.maxTexture3D[0]);
    device_properties.maxTexture3D[1] = static_cast<unsigned>(cudevprops.maxTexture3D[1]);
    device_properties.maxTexture3D[2] = static_cast<unsigned>(cudevprops.maxTexture3D[2]);
    device_properties.maxTexture3DAlt[0] = static_cast<unsigned>(cudevprops.maxTexture3DAlt[0]);
    device_properties.maxTexture3DAlt[1] = static_cast<unsigned>(cudevprops.maxTexture3DAlt[1]);
    device_properties.maxTexture3DAlt[2] = static_cast<unsigned>(cudevprops.maxTexture3DAlt[2]);
    device_properties.maxTextureCubemap = static_cast<unsigned>(cudevprops.maxTextureCubemap);
    device_properties.maxTexture1DLayered[0] = static_cast<unsigned>(cudevprops.maxTexture1DLayered[0]);
    device_properties.maxTexture1DLayered[1] = static_cast<unsigned>(cudevprops.maxTexture1DLayered[1]);
    device_properties.maxTexture2DLayered[0] = static_cast<unsigned>(cudevprops.maxTexture2DLayered[0]);
    device_properties.maxTexture2DLayered[1] = static_cast<unsigned>(cudevprops.maxTexture2DLayered[1]);
    device_properties.maxTexture2DLayered[2] = static_cast<unsigned>(cudevprops.maxTexture2DLayered[2]);
    device_properties.maxTextureCubemapLayered[0] = static_cast<unsigned>(cudevprops.maxTextureCubemapLayered[0]);
    device_properties.maxTextureCubemapLayered[1] = static_cast<unsigned>(cudevprops.maxTextureCubemapLayered[1]);
    device_properties.maxSurface1D = static_cast<unsigned>(cudevprops.maxSurface1D);
    device_properties.maxSurface2D[0] = static_cast<unsigned>(cudevprops.maxSurface2D[0]);
    device_properties.maxSurface2D[1] = static_cast<unsigned>(cudevprops.maxSurface2D[1]);
    device_properties.maxSurface3D[0] = static_cast<unsigned>(cudevprops.maxSurface3D[0]);
    device_properties.maxSurface3D[1] = static_cast<unsigned>(cudevprops.maxSurface3D[1]);
    device_properties.maxSurface3D[2] = static_cast<unsigned>(cudevprops.maxSurface3D[2]);
    device_properties.maxSurface1DLayered[0] = static_cast<unsigned>(cudevprops.maxSurface1DLayered[0]);
    device_properties.maxSurface1DLayered[1] = static_cast<unsigned>(cudevprops.maxSurface1DLayered[1]);
    device_properties.maxSurface2DLayered[0] = static_cast<unsigned>(cudevprops.maxSurface2DLayered[0]);
    device_properties.maxSurface2DLayered[1] = static_cast<unsigned>(cudevprops.maxSurface2DLayered[1]);
    device_properties.maxSurface2DLayered[2] = static_cast<unsigned>(cudevprops.maxSurface2DLayered[2]);
    device_properties.maxSurfaceCubemap = static_cast<unsigned>(cudevprops.maxSurfaceCubemap);
    device_properties.maxSurfaceCubemapLayered[0] = static_cast<unsigned>(cudevprops.maxSurfaceCubemapLayered[0]);
    device_properties.maxSurfaceCubemapLayered[1] = static_cast<unsigned>(cudevprops.maxSurfaceCubemapLayered[1]);
    device_properties.surfaceAlignment = static_cast<unsigned>(cudevprops.surfaceAlignment);
    device_properties.concurrentKernels = static_cast<unsigned>(cudevprops.concurrentKernels);
    device_properties.ECCEnabled = static_cast<unsigned>(cudevprops.ECCEnabled);
    device_properties.pciBusID = static_cast<unsigned>(cudevprops.pciBusID);
    device_properties.pciDeviceID = static_cast<unsigned>(cudevprops.pciDeviceID);
    device_properties.pciDomainID = static_cast<unsigned>(cudevprops.pciDomainID);
    device_properties.tccDriver = static_cast<unsigned>(cudevprops.tccDriver);
    device_properties.asyncEngineCount = static_cast<unsigned>(cudevprops.asyncEngineCount);
    device_properties.unifiedAddressing = static_cast<unsigned>(cudevprops.unifiedAddressing);
    device_properties.memoryClockRate = static_cast<unsigned>(cudevprops.memoryClockRate);
    device_properties.memoryBusWidth = static_cast<unsigned>(cudevprops.memoryBusWidth);
    device_properties.l2CacheSize = static_cast<unsigned>(cudevprops.l2CacheSize);
    device_properties.persistingL2CacheMaxSize = static_cast<unsigned>(cudevprops.persistingL2CacheMaxSize);
    device_properties.maxThreadsPerMultiProcessor = static_cast<unsigned>(cudevprops.maxThreadsPerMultiProcessor);
    device_properties.streamPrioritiesSupported = static_cast<unsigned>(cudevprops.streamPrioritiesSupported);
    device_properties.globalL1CacheSupported = static_cast<unsigned>(cudevprops.globalL1CacheSupported);
    device_properties.localL1CacheSupported = static_cast<unsigned>(cudevprops.localL1CacheSupported);
    device_properties.sharedMemPerMultiprocessor = static_cast<unsigned>(cudevprops.sharedMemPerMultiprocessor);
    device_properties.regsPerMultiprocessor = static_cast<unsigned>(cudevprops.regsPerMultiprocessor);
    device_properties.managedMemory = static_cast<unsigned>(cudevprops.managedMemory);
    device_properties.isMultiGpuBoard = static_cast<unsigned>(cudevprops.isMultiGpuBoard);
    device_properties.multiGpuBoardGroupID = static_cast<unsigned>(cudevprops.multiGpuBoardGroupID);
    device_properties.singleToDoublePrecisionPerfRatio = static_cast<unsigned>(cudevprops.singleToDoublePrecisionPerfRatio);
    device_properties.pageableMemoryAccess = static_cast<unsigned>(cudevprops.pageableMemoryAccess);
    device_properties.concurrentManagedAccess = static_cast<unsigned>(cudevprops.concurrentManagedAccess);
    device_properties.computePreemptionSupported = static_cast<unsigned>(cudevprops.computePreemptionSupported);
    device_properties.canUseHostPointerForRegisteredMem = static_cast<unsigned>(cudevprops.canUseHostPointerForRegisteredMem);
    device_properties.cooperativeLaunch = static_cast<unsigned>(cudevprops.cooperativeLaunch);
    device_properties.cooperativeMultiDeviceLaunch = static_cast<unsigned>(cudevprops.cooperativeMultiDeviceLaunch);
    device_properties.pageableMemoryAccessUsesHostPageTables = static_cast<unsigned>(cudevprops.pageableMemoryAccessUsesHostPageTables);
    device_properties.directManagedMemAccessFromHost = static_cast<unsigned>(cudevprops.directManagedMemAccessFromHost);
    device_properties.accessPolicyMaxWindowSize = static_cast<unsigned>(cudevprops.accessPolicyMaxWindowSize);
}

void
cudapp::DeviceProperties::pretty_print() {
    // now cast and translate
    cout << "name = " << device_properties.name << endl;
    //cout<<device_propertiesuuid; // 16 byte unique identifier
    cout << "totalGlobalMem = " << device_properties.totalGlobalMem << endl;
    cout << "sharedMemPerBlock = " << device_properties.sharedMemPerBlock << endl;
    cout << "regsPerBlock = " << device_properties.regsPerBlock << endl;
    cout << "warpSize = " << device_properties.warpSize << endl;
    cout << "memPitch = " << device_properties.memPitch << endl;
    cout << "maxThreadsPerBlock = " << device_properties.maxThreadsPerBlock << endl;
    cout << "maxThreadsDim = ("
         << device_properties.maxThreadsDim[0] << ","
         << device_properties.maxThreadsDim[1]
         << ")" << endl;
    cout << "maxGridSize = ("
         << device_properties.maxGridSize[0] << ","
         << device_properties.maxGridSize[1]
         << ")" << endl;
    cout << "clockRate = " << device_properties.clockRate << endl;
    cout << "totalConstMem = " << device_properties.totalConstMem << endl;
    cout << "major = " << device_properties.major << endl;
    cout << "minor = " << device_properties.minor << endl;
    cout << "textureAlignment = " << device_properties.textureAlignment << endl;
    cout << "texturePitchAlignment = " << device_properties.texturePitchAlignment << endl;
    cout << "deviceOverlap = " << device_properties.deviceOverlap << endl;
    cout << "multiProcessorCount = " << device_properties.multiProcessorCount << endl;
    cout << "kernelExecTimeoutEnabled = " << device_properties.kernelExecTimeoutEnabled << endl;
    cout << "integrated = " << device_properties.integrated << endl;
    cout << "canMapHostMemory = " << device_properties.canMapHostMemory << endl;
    cout << "computeMode = " << device_properties.computeMode << endl;
    cout << "maxTexture1D = " << device_properties.maxTexture1D << endl;
    cout << "maxTexture1DMipmap = " << device_properties.maxTexture1DMipmap << endl;
    cout << "maxTexture1DLinear = " << device_properties.maxTexture1DLinear << endl;
    cout << "maxTexture2D = ("
         << device_properties.maxTexture2D[0] << ","
         << device_properties.maxTexture2D[1]
         << ")" << endl;
    cout << "maxTexture2DMipmap = ("
         << device_properties.maxTexture2DMipmap[0] << ","
         << device_properties.maxTexture2DMipmap[1]
         << ")" << endl;
    cout << "maxTexture2DLinear = ("
         << device_properties.maxTexture2DLinear[0] << ","
         << device_properties.maxTexture2DLinear[1] << ","
         << device_properties.maxTexture2DLinear[2]
         << ")" << endl;
    cout << "maxTexture2DGather = ("
         << device_properties.maxTexture2DGather[0] << ","
         << device_properties.maxTexture2DGather[1]
         << ")" << endl;
    cout << "maxTexture3D = ("
         << device_properties.maxTexture3D[0] << ","
         << device_properties.maxTexture3D[1] << ","
         << device_properties.maxTexture3D[2]
         << ")" << endl;
    cout << "maxTexture3DAlt = ("
         << device_properties.maxTexture3DAlt[0] << ","
         << device_properties.maxTexture3DAlt[1] << ","
         << device_properties.maxTexture3DAlt[2]
         << ")" << endl;
    cout << "maxTextureCubemap = " << device_properties.maxTextureCubemap << endl;
    cout << "maxTexture1DLayered = ("
         << device_properties.maxTexture1DLayered[0] << ","
         << device_properties.maxTexture1DLayered[1]
         << ")" << endl;
    cout << "maxTexture2DLayered = ("
         << device_properties.maxTexture2DLayered[0] << ","
         << device_properties.maxTexture2DLayered[1] << ","
         << device_properties.maxTexture2DLayered[2]
         << ")" << endl;
    cout << "maxTextureCubemapLayered = ("
         << device_properties.maxTextureCubemapLayered[0] << ","
         << device_properties.maxTextureCubemapLayered[1]
         << ")" << endl;
    cout << "maxSurface1D = " << device_properties.maxSurface1D << endl;
    cout << "maxSurface2D = ("
         << device_properties.maxSurface2D[0] << ","
         << device_properties.maxSurface2D[1]
         << ")" << endl;
    cout << "maxSurface3D = ("
         << device_properties.maxSurface3D[0] << ","
         << device_properties.maxSurface3D[1] << ","
         << device_properties.maxSurface3D[2]
         << ")" << endl;
    cout << "maxSurface1DLayered = ("
         << device_properties.maxSurface1DLayered[0] << ","
         << device_properties.maxSurface1DLayered[1]
         << ")" << endl;
    cout << "maxSurface2DLayered = ("
         << device_properties.maxSurface2DLayered[0] << ","
         << device_properties.maxSurface2DLayered[1] << ","
         << device_properties.maxSurface2DLayered[2]
         << ")" << endl;
    cout << "maxSurfaceCubemap = " << device_properties.maxSurfaceCubemap << endl;
    cout << "maxSurfaceCubemapLayered = ("
         << device_properties.maxSurfaceCubemapLayered[0] << ","
         << device_properties.maxSurfaceCubemapLayered[1]
         << ")" << endl;
    cout << "surfaceAlignment = " << device_properties.surfaceAlignment << endl;
    cout << "concurrentKernels = " << device_properties.concurrentKernels << endl;
    cout << "ECCEnabled = " << device_properties.ECCEnabled << endl;
    cout << "pciBusID = " << device_properties.pciBusID << endl;
    cout << "pciDeviceID = " << device_properties.pciDeviceID << endl;
    cout << "pciDomainID = " << device_properties.pciDomainID << endl;
    cout << "tccDriver = " << device_properties.tccDriver << endl;
    cout << "asyncEngineCount = " << device_properties.asyncEngineCount << endl;
    cout << "unifiedAddressing = " << device_properties.unifiedAddressing << endl;
    cout << "memoryClockRate = " << device_properties.memoryClockRate << endl;
    cout << "memoryBusWidth = " << device_properties.memoryBusWidth << endl;
    cout << "l2CacheSize = " << device_properties.l2CacheSize << endl;
    cout << "persistingL2CacheMaxSize = " << device_properties.persistingL2CacheMaxSize << endl;
    cout << "maxThreadsPerMultiProcessor = " << device_properties.maxThreadsPerMultiProcessor << endl;
    cout << "streamPrioritiesSupported = " << device_properties.streamPrioritiesSupported << endl;
    cout << "globalL1CacheSupported = " << device_properties.globalL1CacheSupported << endl;
    cout << "localL1CacheSupported = " << device_properties.localL1CacheSupported << endl;
    cout << "sharedMemPerMultiprocessor = " << device_properties.sharedMemPerMultiprocessor << endl;
    cout << "regsPerMultiprocessor = " << device_properties.regsPerMultiprocessor << endl;
    cout << "managedMemory = " << device_properties.managedMemory << endl;
    cout << "isMultiGpuBoard = " << device_properties.isMultiGpuBoard << endl;
    cout << "multiGpuBoardGroupID = " << device_properties.multiGpuBoardGroupID << endl;
    cout << "singleToDoublePrecisionPerfRatio = " << device_properties.singleToDoublePrecisionPerfRatio
         << endl;
    cout << "pageableMemoryAccess = " << device_properties.pageableMemoryAccess << endl;
    cout << "concurrentManagedAccess = " << device_properties.concurrentManagedAccess << endl;
    cout << "computePreemptionSupported = " << device_properties.computePreemptionSupported << endl;
    cout << "canUseHostPointerForRegisteredMem = "
         << device_properties.canUseHostPointerForRegisteredMem << endl;
    cout << "cooperativeLaunch = " << device_properties.cooperativeLaunch << endl;
    cout << "cooperativeMultiDeviceLaunch = " << device_properties.cooperativeMultiDeviceLaunch
         << endl;
    cout << "pageableMemoryAccessUsesHostPageTables = "
         << device_properties.pageableMemoryAccessUsesHostPageTables << endl;
    cout << "directManagedMemAccessFromHost = " << device_properties.directManagedMemAccessFromHost
         << endl;
    cout << "accessPolicyMaxWindowSize = " << device_properties.accessPolicyMaxWindowSize << endl;

}


cudapp::DeviceProperties::~DeviceProperties() {}

