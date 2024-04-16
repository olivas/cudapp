//
// Created by olivas on 4/15/24.
//

#include <iostream>
#include <memory>
#include <cudapp/device_manager.hpp>

using std::cerr;
using std::endl;

using cudapp::DeviceManager;

int main(int argc, char** argv){
    std::shared_ptr <DeviceManager> device_mgr(new DeviceManager);

    auto device_count = device_mgr->device_count();
    if(device_count){
        cerr<<"device_count = "<<*device_count<<endl;
    }

    auto current_device = device_mgr->current_device();
    if(current_device){
        cerr<<"current_device = "<<*current_device<<endl;
    }

}




