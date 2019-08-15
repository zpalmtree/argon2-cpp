# Add the files we want to link against
set(cross_platform_intrinsics_source_files
    BlakeIntrinsics.cpp
)

# Add the library to be linked against, with the previously specified source files
add_library(CrossPlatformIntrinsics ${cross_platform_intrinsics_source_files})
