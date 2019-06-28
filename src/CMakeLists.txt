# Add the blake2 directory to add that library
add_subdirectory(Blake2)

# Add the Blake2 directory as an include directory, so we can #include <Blake2/Blake2b.h>
target_include_directories(Blake2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
