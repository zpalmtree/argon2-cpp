# Add the files we want to link against
set(blake2_source_files
    Blake2b.cpp
)

# Add the library to be linked against, with the previously specified source files
add_library(Blake2 ${blake2_source_files})

# Use the TargetArch module to figure out what arch we're compiling for
include(TargetArch)
target_architecture(ARCH)

message(STATUS "Target Architecture: ${ARCH}")

if("${ARCH}" STREQUAL "x86_64")
    message(STATUS "Using optimized Blake2 code")
    target_link_libraries(Blake2 X86Intrinsics)
elseif()
    message(STATUS "Using crossplatform code")
    target_link_libraries(Blake2 CrossPlatformIntrinsics)
endif()
