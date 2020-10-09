# Add the files we want to link against
set(argon_nvidia_source_files
    Argon2.cu
    Blake2.cu
)

# Add the library to be linked against, with the previously specified source files
add_library(Nvidia ${argon_nvidia_source_files})
