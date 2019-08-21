# Add the Argon2 directory to add that library
add_subdirectory(Argon2)

# Add the Blake2 directory to add that library
add_subdirectory(Blake2)

# Use the TargetArch module to figure out what arch we're compiling for
include(TargetArch)
target_architecture(ARCH)

message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# For some reason, CMAKE_SYSTEM_PROCESSOR does not appear to remain set when
# cross compiling, so we get into cpu_features, and it tries to include the x86_64
# version. I'm just gonna set it manually rather than figure out what's going wrong.
if("${ARCH}" STREQUAL "armv8" OR "${ARCH}" STREQUAL "armv7" OR "${ARCH}" STREQUAL "arm")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
endif()

# Add cpu_features if we're going to need them for runtime dispatching
if("${ARCH}" STREQUAL "x86_64" OR "${ARCH}" STREQUAL "armv8" OR "${ARCH}" STREQUAL "armv7" OR "${ARCH}" STREQUAL "arm")
    add_subdirectory(cpu_features)
endif()

add_subdirectory(Intrinsics)

# Add the Argon2 directory as an include directory, so we can #include <Argon2/Argon2.h>
target_include_directories(Argon2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Add the Blake2 directory as an include directory, so we can #include <Blake2/Blake2b.h>
target_include_directories(Blake2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
