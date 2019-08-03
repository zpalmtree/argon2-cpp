# We need 3.8 for set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.8)

project(argon2-cpp)

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

add_subdirectory(src)
add_subdirectory(tests)
