# Add an executable called test with main.cpp as the entrypoint
add_executable(test main.cpp)

# Link test to the Argon2 and Blake2 library
target_link_libraries(test Argon2 Blake2)
