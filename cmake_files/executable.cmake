
add_executable(hello_device examples/hello_device.cpp)
target_link_libraries(hello_device cudapp)

add_executable(hello_quda examples/hello_quda.cu)
target_link_libraries(hello_quda cudapp
        /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so
        /usr/lib/x86_64-linux-gnu/libcuquantum/12/libcutensornet.so
        /usr/lib/x86_64-linux-gnu/libcuquantum/12/libcustatevec.so
)

set_property(TARGET hello_quda PROPERTY CUDA_ARCHITECTURES OFF)