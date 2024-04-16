#include <cudapp/device_manager.hpp>
#include <cudapp/check_error.cuh>

using std::optional;

cudapp::DeviceManager::DeviceManager() {}

optional<unsigned>
cudapp::DeviceManager::device_count() const {
    int device_count{0};
    CHECK_ERROR(cudaGetDeviceCount(&device_count));
    return static_cast<unsigned>(device_count);
}

optional<unsigned>
cudapp::DeviceManager::current_device() const {
    int device{0};
    CHECK_ERROR(cudaGetDevice(&device));
    return static_cast<unsigned>(device);
}

cudapp::DeviceManager::~DeviceManager() {}

