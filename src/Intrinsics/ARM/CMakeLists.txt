set(neon_intrinsics_source_files
    CompressNEON.cpp
    ProcessBlockNEON.cpp
)

# Add the files we want to link against
set(arm_intrinsics_source_files
    ArgonIntrinsics.cpp
    BlakeIntrinsics.cpp
    ${neon_intrinsics_source_files}
)

# Add the library to be linked against, with the previously specified source files
add_library(Intrinsics ${arm_intrinsics_source_files})

target_link_libraries(Intrinsics cpu_features)
