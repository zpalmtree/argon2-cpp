// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include "Argon2/Argon2.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace ProcessBlockSSSE3
{
    void blamkaG1SSSE3(
        __m128i& a0, __m128i& a1, __m128i& b0, __m128i& b1,
        __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void blamkaG2SSSE3(
        __m128i& a0, __m128i& a1, __m128i& b0, __m128i& b1,
        __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void diagonalizeSSSE3(
        __m128i& b0, __m128i& b1, __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void undiagonalizeSSSE3(
        __m128i& b0, __m128i& b1, __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1);

    void processBlockSSSE3(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor);
}
