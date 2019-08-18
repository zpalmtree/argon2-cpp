// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>
#include <vector>

#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace CompressAVX2
{
    void g1AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]);

    void g2AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]);

    void diagonalizeAVX2(__m256i& b, __m256i& c, __m256i& d);

    void undiagonalizeAVX2(__m256i& b, __m256i& c, __m256i& d);

    void compressAVX2(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags);
}
