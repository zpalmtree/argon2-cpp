# Add the Argon2 directory to add that library
add_subdirectory(Argon2)

# Add the Blake2 directory to add that library
add_subdirectory(Blake2)

# Use the TargetArch module to figure out what arch we're compiling for
include(TargetArch)
target_architecture(ARCH)

if("${ARCH}" STREQUAL "x86_64")
    add_subdirectory(cpu_features)
endif()

add_subdirectory(Intrinsics)

# Add the Argon2 directory as an include directory, so we can #include <Argon2/Argon2.h>
target_include_directories(Argon2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Add the Blake2 directory as an include directory, so we can #include <Blake2/Blake2b.h>
target_include_directories(Blake2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
