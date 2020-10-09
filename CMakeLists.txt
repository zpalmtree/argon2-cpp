# We need 3.8 for set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.8)

project(argon2-cpp LANGUAGES CXX)

option(ENABLE_NVIDIA "Enable Nvidia support" ON)
option(NVIDIA_ARCH "Nvidia architectures to compile for" "10 13 20 30 35 37 50 52 60 61 70 72 75")

if (ENABLE_NVIDIA)
    enable_language(CUDA)
endif()

# Assert our compiler is good
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC 7.0 or higher
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
        message(FATAL_ERROR "GCC/G++ 7.0 or greater is required to compile. Please check the README.md for detailed compilation instructions.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Clang 6.0 or higher
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6)
        message(FATAL_ERROR "Clang 6.0 or greater is required to compile. Please check the README.md for detailed compilation instructions.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Visual Studio 15 2017 or higher
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
        message(FATAL_ERROR "MSVC 19.10 or greater is required to compile (Latest Visual Studio 15 2017 should suffice). Please check the README.md for detailed compilation instructions.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
else()
    message(WARNING "You are using an unsupported compiler. The compilation is likely to fail. God speed.")
endif()

# Enable c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pop binaries in the root build folder, not nested
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Set optimization on by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Allow people using library to override this option with SET(BUILD_TESTS OFF)
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

include(CheckCXXCompilerFlag)

# GOTTA GO FAST
if (NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Add custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/Intrinsics")

option(BUILD_TESTS "Build the test executable?" ON)

add_subdirectory(src)

if (${BUILD_TESTS})
    add_subdirectory(tests)
endif()

if (ENABLE_NVIDIA)
    list(APPEND NVIDIA_TARGETS Nvidia)

    foreach(target ${NVIDIA_TARGETS})
        message(STATUS ${target})
        target_compile_features(${target} PUBLIC cxx_std_11)
    endforeach()

    set_target_properties(${NVIDIA_TARGETS}
                          PROPERTIES
                          CUDA_ARCHITECTURES "${NVIDIA_ARCH}"
                          CUDA_SEPARABLE_COMPILATION ON
                          CUDA_STANDARD 11)

    if (MSVC)
        set_target_properties(${NVIDIA_TARGETS}
                              PROPERTIES
                              CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()

    # Makes cuda-memcheck, cuda-gdb, etc, much more helpful
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    endif()
endif()
