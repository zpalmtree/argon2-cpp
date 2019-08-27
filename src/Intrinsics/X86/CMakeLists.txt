set(avx512_source_files
    ArgonIntrinsics.cpp
    CompressAVX512.cpp
    ProcessBlockAVX512.cpp
)

set(avx2_source_files
    CompressAVX2.cpp
    ProcessBlockAVX2.cpp
)

set(sse41_source_files
    CompressSSE41.cpp
)

set(ssse3_source_files
    CompressSSSE3.cpp
    ProcessBlockSSSE3.cpp
)

set(sse2_source_files
    CompressSSE2.cpp
    ProcessBlockSSE2.cpp
)

set(x86_intrinsics_source_files
    BlakeIntrinsics.cpp
    ${avx512_source_files}
    ${avx2_source_files}
    ${sse41_source_files}
    ${ssse3_source_files}
    ${sse2_source_files}
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")

    message(STATUS "Enabling GNU style -mavx flags")

    set_source_files_properties(${avx512_source_files} PROPERTIES COMPILE_FLAGS -mavx512f)

    set_source_files_properties(${avx2_source_files} PROPERTIES COMPILE_FLAGS -mavx2)

    set_source_files_properties(${sse41_source_files} PROPERTIES COMPILE_FLAGS -msse4.1)

    set_source_files_properties(${ssse3_source_files} PROPERTIES COMPILE_FLAGS -mssse3)

    set_source_files_properties(${sse2_source_files} PROPERTIES COMPILE_FLAGS -msse2)

elseif(MSVC AND NOT MSVC_VERSION LESS 1800)

    message(STATUS "Enabling Windows style /arch:AVX flags")

    set_source_files_properties(${avx2_source_files} PROPERTIES COMPILE_FLAGS /arch:AVX2)

else()
    message(STATUS "Compile flags not needed to enable intrinsics")
endif()

# Add the library to be linked against, with the previously specified source files
add_library(Intrinsics ${x86_intrinsics_source_files})

target_link_libraries(Intrinsics cpu_features)
