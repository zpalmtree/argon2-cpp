// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>
#include <vector>

#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace CompressSSSE3
{
    void g1SSSE3(
        __m128i& row1l, __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row1h, __m128i& row2h, __m128i& row3h, __m128i& row4h,
        __m128i& b0, __m128i& b1);

    void g2SSSE3(
        __m128i& row1l, __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row1h, __m128i& row2h, __m128i& row3h, __m128i& row4h,
        __m128i& b0, __m128i& b1);

    void diagonalizeSSSE3(
        __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row2h, __m128i& row3h, __m128i& row4h);

    void undiagonalizeSSSE3(
        __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row2h, __m128i& row3h, __m128i& row4h);

    void Round(
        int r,
        const uint64_t* block_ptr,
        __m128i& row1l, __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row1h, __m128i& row2h, __m128i& row3h, __m128i& row4h);

    void compressSSSE3(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags);
}
