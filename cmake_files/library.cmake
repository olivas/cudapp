
add_library(cudapp SHARED
        src/device.cu
        src/device_manager.cu
)
set_property(TARGET cudapp PROPERTY CUDA_ARCHITECTURES OFF)

