# cudapp
C++ API to CUDA C Runtime

## Rationale
The idea behind this project is to provide safety mechanisms and patterns inherent to C++
and to provide a more natural interface to those who have become more comfortable with C++.
If you have a C++ project and you want to use CUDA, you'll probably end up implementing a
lot of this boilerplate code already.

### RAII/CADRe
This is the first C++ pattern that jumps out with a clear use case in CUDA.

For the unfamiliar RAII stands for Resource Allocation Is Initialization.  I prefer 
Constructor Allocates, Destructor Releases.  In CUDA memory management, as in most 
C libraries, for every cudaMalloc you have to call cudaFree once and only once.  The
pattern is simple - call cudaMalloc in an object's constructor and cudaFree in the 
object's destructor.  This way you're guaranteed to pair frees with mallocs and never 
have to worry about attempting to access memory that's already been freed.  If the 
object is in scope then the resource is valid.

### Types
This may be a common C pattern, but I see a gratuitous use of 'int' when other types
might be more appropriate, e.g. bool and unsigned.  One potential reason to use int is 
to indicate failure or "not set."  I prefer std::optional<>.

## Getting Started

First step is to install [CUDA](https://developer.nvidia.com/cuda-downloads) and
then [cmake](https://cmake.org/download/).

Those are the only dependencies beyond the standard C++ toolchain.

### Build
The checkout and build, including running tests, is pretty standard:
```bash
git clone git@github.com:olivas/cudapp.git
cd cudapp
mkdir build
cd build
cmake ..
make 
make test
```
You should see something similar to the following output: 

    Running tests...
    Test project /home/olivas/cudapp/build
        Start 1: test_device
    1/3 Test #1: test_device ......................   Passed    0.35 sec
        Start 2: test_device_manager
    2/3 Test #2: test_device_manager ..............   Passed    0.09 sec
        Start 3: test_device_properties
    3/3 Test #3: test_device_properties ...........   Passed    0.08 sec
    
    100% tests passed, 0 tests failed out of 3
    
    Total Test time (real) =   0.53 sec

### Next Steps

#### The Basics
The first thing you might want to do is see how many NVIDIA cards you have installed and check their properties.



