//
// Created by olivas on 4/15/24.
//
#include <cassert>

#include <iostream>
#include <memory>

#include <cudapp/device.hpp>
#include <cudapp/device_properties.hpp>
#include <cudapp/device_manager.hpp>

using std::cerr;
using std::endl;

using cudapp::Device;
using cudapp::DeviceManager;
using cudapp::DeviceProperties;

int main(int argc, char *argv[]) {

    unsigned device_number{0};
    if(argc == 2){
        device_number = static_cast<unsigned>(atoi(argv[1]));
    }

    std::shared_ptr <Device> device(new Device(1));
    std::cout<<device->multi_processor_count()<<std::endl;
    std::cout<<device->total_global_mem()<<std::endl;

    std::shared_ptr <DeviceManager> device_mgr(new DeviceManager);

    auto device_count = device_mgr->device_count();
    if(device_count){
        cerr<<"device_count = "<<*device_count<<endl;
    }

    auto current_device = device_mgr->current_device();
    if(current_device){
        cerr<<"current_device = "<<*current_device<<endl;
    }

    DeviceProperties device_properties(device_number);
    device_properties.pretty_print();

}




