cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(clib LANGUAGES CXX C)

find_package(Python REQUIRED)
# find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# modify to your own python path, or delete it if ok
include_directories(/home/fx/lib/anaconda/include/python3.11/)
include_directories(./)
include_directories(/home/fx/lib/xnnpack/include/)

set(SRCS sdpa_wrapper.cpp)
add_library(clib SHARED ${SRCS})

target_link_libraries(clib
	/home/fx/project/conf24/csrc/libmeformer.a
	/home/fx/lib/openblas_s/lib/libopenblas.a

	/home/fx/lib/xnnpack/lib64/libXNNPACK.a
	/home/fx/lib/xnnpack/lib64/libcpuinfo.a
	/home/fx/lib/xnnpack/lib64/libpthreadpool.a

	"${TORCH_LIBRARIES}"
	"${TORCH_PYTHON_LIBRARY}"
)
