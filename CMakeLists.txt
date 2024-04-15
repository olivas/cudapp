cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(cudapp LANGUAGES CXX CUDA)

include(cmake_files/configuration.cmake)
include(cmake_files/tests.cmake)
include(cmake_files/library.cmake)
include(cmake_files/executable.cmake)
