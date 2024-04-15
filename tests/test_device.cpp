//
// Created by olivas on 4/15/24.
//
#include <cassert>

#include <iostream>
#include <memory>

#include <cudapp/device.hpp>

using std::cerr;
using std::endl;

int main(int argc, char *argv[]) {

    std::shared_ptr <Device> device(new Device(1));
    std::cout<<device->multi_processor_count()<<std::endl;
    std::cout<<device->total_global_mem()<<std::endl;
}




