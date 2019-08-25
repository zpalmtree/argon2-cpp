// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>
#include <vector>

#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace CompressAVX2
{
    #if _MSC_VER && !__INTEL_COMPILER
    #pragma optimize("", off)
    #endif
    inline void loadChunk(
        __m256i &m0, __m256i &m1, __m256i &m2, __m256i &m3,
        __m256i &m4, __m256i &m5, __m256i &m6, __m256i &m7,
        std::vector<uint64_t> &chunk)
    {
        m0 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[0])));
        m1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[2])));
        m2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[4])));
        m3 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[6])));
        m4 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[8])));
        m5 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[10])));
        m6 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[12])));
        m7 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[14])));
    }

    void g1AVX2(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& m);

    void g2AVX2(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& m);

    void diagonalizeAVX2(__m256i& a, __m256i& c, __m256i& d);

    void undiagonalizeAVX2(__m256i& a, __m256i& c, __m256i& d);

    void compressAVX2(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags);
}
