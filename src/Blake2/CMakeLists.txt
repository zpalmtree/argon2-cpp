# Add the files we want to link against
set(blake2_source_files
    Blake2b.cpp
)

# Add the library to be linked against, with the previously specified source files
add_library(Blake2 ${blake2_source_files})

target_link_libraries(Blake2 Intrinsics)
