# Use the TargetArch module to figure out what arch we're compiling for
include(TargetArch)
target_architecture(ARCH)

message(STATUS "Target Architecture: ${ARCH}")

if("${ARCH}" STREQUAL "x86_64")
    message(STATUS "Using x86 specific code")
    add_subdirectory(X86)
elseif("${ARCH}" STREQUAL "arm" OR "${ARCH}" STREQUAL "armv8" OR "${ARCH}" STREQUAL "armv7")
    message(STATUS "Using ARM specific code")
    add_subdirectory(ARM)
else()
    message(STATUS "Using cross platform code")
    add_subdirectory(CrossPlatform)
endif()
