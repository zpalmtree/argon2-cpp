// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#if defined(_MSC_VER)
    #pragma message("Including <intrin.h>")
    /* Microsoft C/C++-compatible compiler */
    #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    #pragma message("Including <x86intrin.h>")
    /* GCC-compatible compiler, targeting x86/x86-64 */
    #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
    #pragma message("Including <mmintrin.h>")
    /* GCC-compatible compiler, targeting ARM with WMMX */
    #include <mmintrin.h>
#else
    #error Your compiler does not support <x86intrin.h>. Please upgrade to one that does.
#endif
