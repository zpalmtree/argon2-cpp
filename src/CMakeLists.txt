# Add the Argon2 directory to add that library
add_subdirectory(Argon2)

# Add the Blake2 directory to add that library
add_subdirectory(Blake2)

# Add the Utilities directory to add that library
add_subdirectory(Utilities)

# Add the Argon2 directory as an include directory, so we can #include <Argon2/Argon2.h>
target_include_directories(Argon2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Add the Blake2 directory as an include directory, so we can #include <Blake2/Blake2b.h>
target_include_directories(Blake2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Add the Utilities directory as an include directory, so we can #include <Utilities/Utilities.h>
target_include_directories(Utilities PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
