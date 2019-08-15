# Use the TargetArch module to figure out what arch we're compiling for
include(TargetArch)
target_architecture(ARCH)

message(STATUS "Target Architecture: ${ARCH}")

if("${ARCH}" STREQUAL "x86_64")
    message(STATUS "Using x86 specific code")
    add_subdirectory(X86)
elseif()
    message(STATUS "Using cross platform code")
    add_subdirectory(CrossPlatform)
endif()
