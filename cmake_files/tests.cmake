
enable_testing()

add_executable(test_device tests/test_device.cpp)
target_link_libraries(test_device cudapp)
add_test(test_device bin/test_device)

add_executable(test_device_manager tests/test_device_manager.cpp)
target_link_libraries(test_device_manager cudapp)
add_test(test_device_manager bin/test_device_manager)
