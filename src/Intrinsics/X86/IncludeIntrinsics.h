// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#if defined(_MSC_VER)
    /* Microsoft C/C++-compatible compiler */
    #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    /* GCC-compatible compiler, targeting x86/x86-64 */
    #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
    /* GCC-compatible compiler, targeting ARM with WMMX */
    #include <mmintrin.h>
#else
    #error Your compiler does not support <x86intrin.h>. Please upgrade to one that does.
#endif
