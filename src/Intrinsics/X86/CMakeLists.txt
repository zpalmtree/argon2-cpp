# Add the files we want to link against
set(x86_intrinsics_source_files
    BlakeIntrinsics.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Enabling -mavx2 flag")
    set_source_files_properties(${x86_intrinsics_source_files} PROPERTIES COMPILE_FLAGS -mavx2)
elseif(MSVC AND NOT MSVC_VERSION LESS 1800)
    message(STATUS "ENABLING /arch:AVX2 flag")
    set_source_files_properties(${x86_intrinsics_source_files} PROPERTIES COMPILE_FLAGS /arch:AVX2)
else()
    message(STATUS "AVX2 compile flags not needed")
endif()

# Add the library to be linked against, with the previously specified source files
add_library(X86Intrinsics ${x86_intrinsics_source_files})

target_link_libraries(X86Intrinsics cpu_features)