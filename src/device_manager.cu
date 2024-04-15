#include <cudapp/device_manager.hpp>
#include <cudapp/check_error.cuh>

DeviceManager::DeviceManager()
{
  CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceBlockingSync));
  CHECK_ERROR(cudaSetDevice(device_number_));  
  std::cerr<<"device_number_ = "<<device_number_<<std::endl;

  cudaDeviceProp device_properties;
  CHECK_ERROR(cudaGetDeviceProperties(&device_properties, device_number_));
  total_global_mem_ = device_properties.totalGlobalMem;
  multi_processor_count_ = device_properties.multiProcessorCount;  
}

DeviceManager::~DeviceManager()
{  
  CHECK_ERROR(cudaDeviceReset());
}

