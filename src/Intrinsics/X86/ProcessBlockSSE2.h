// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include "Argon2/Argon2.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace ProcessBlockSSE2
{
    void blamkaG1SSE2(
        __m128i& a0, __m128i& a1, __m128i& b0, __m128i& b1,
        __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void blamkaG2SSE2(
        __m128i& a0, __m128i& a1, __m128i& b0, __m128i& b1,
        __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void diagonalizeSSE2(
        __m128i& b0, __m128i& b1, __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void undiagonalizeSSE2(
        __m128i& b0, __m128i& b1, __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void processBlockSSE2(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor);
}
