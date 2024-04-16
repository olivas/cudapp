//
// Created by olivas on 4/15/24.
//

#include <iostream>
#include <cudapp/device_properties.hpp>

using std::cerr;
using std::endl;

using cudapp::DeviceProperties;

int main(int argc, char** argv){
    unsigned device_number{0};
    DeviceProperties device_properties(device_number);
    device_properties.pretty_print();
}




