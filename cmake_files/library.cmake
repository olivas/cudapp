

add_library(cudapp SHARED
        src/device.cu
        src/device_properties.cu
        src/device_manager.cu
        ../include/cudapp/cuda_device_prop.hpp
)
set_property(TARGET cudapp PROPERTY CUDA_ARCHITECTURES OFF)
