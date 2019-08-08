# Add an executable called argon2-cpp-test with main.cpp as the entrypoint
add_executable(argon2-cpp-test main.cpp)

# Link test to the Argon2 and Blake2 library
target_link_libraries(argon2-cpp-test Argon2 Blake2)
