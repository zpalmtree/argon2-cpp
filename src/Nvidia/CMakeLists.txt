# Add the files we want to link against
set(argon_nvidia_source_files
    Argon2.cu
    Blake2.cu
)

# Add the library to be linked against, with the previously specified source files
add_library(Nvidia ${argon_nvidia_source_files})

target_compile_features(Nvidia PUBLIC cxx_std_11)

set_target_properties(Nvidia PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Makes cuda-memcheck, cuda-gdb, etc, much more helpful
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -c 
    --generate-code=arch=compute_35,code=sm_35
    --generate-code=arch=compute_35,code=compute_35
    --generate-code=arch=compute_61,code=sm_61
    --generate-code=arch=compute_61,code=compute_61
    --generate-code=arch=compute_75,code=sm_75
    --generate-code=arch=compute_75,code=compute_75")
