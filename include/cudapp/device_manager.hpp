#pragma once

#include <optional>

namespace cudapp{
    class DeviceManager {
    public:
        DeviceManager();

        ~DeviceManager();

        [[nodiscard("this method's job is to return a value.")]]
        std::optional<unsigned> device_count() const;

        [[nodiscard("this method's job is to return a value.")]]
        std::optional<unsigned> current_device() const;

    };

}