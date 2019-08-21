set(neon_intrinsics_source_files
    ProcessBlockNEON.cpp
)

# Add the files we want to link against
set(arm_intrinsics_source_files
    ArgonIntrinsics.cpp
    BlakeIntrinsics.cpp
    ${neon_intrinsics_source_files}
)

set_source_files_properties(${neon_intrinsics_source_files} PROPERTIES COMPILE_FLAGS -march=armv8)
set_source_files_properties(${neon_intrinsics_source_files} PROPERTIES COMPILE_FLAGS -mfpu=neon-fp-armv8)
set_source_files_properties(${neon_intrinsics_source_files} PROPERTIES COMPILE_FLAGS -mcpu=cortex-a57)

# Add the library to be linked against, with the previously specified source files
add_library(Intrinsics ${arm_intrinsics_source_files})

target_link_libraries(Intrinsics cpu_features)
