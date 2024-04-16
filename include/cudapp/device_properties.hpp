#pragma once

#include <optional>

#include <cudapp/cuda_device_prop.hpp>

namespace cudapp{
    class DeviceProperties {
    public:

        DeviceProperties(unsigned device_number);

        ~DeviceProperties();

        void change_device_number(unsigned device_number);

        cuda_device_prop device_properties;

        void pretty_print();

    private:

        unsigned device_number_;

        void set_device_properties(const unsigned device_number);
    };
}

