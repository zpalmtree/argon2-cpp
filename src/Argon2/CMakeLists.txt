# Add the files we want to link against
set(argon2_source_files
    Argon2.cpp
)

# Add the library to be linked against, with the previously specified source files
add_library(Argon2 ${argon2_source_files})

target_link_libraries(Argon2 Blake2)
