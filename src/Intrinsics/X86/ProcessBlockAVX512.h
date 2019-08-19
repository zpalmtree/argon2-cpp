// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include "Argon2/Argon2.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace ProcessBlockAVX512
{
    inline __m512i muladd(__m512i &x, __m512i &y)
    {
        __m512i z = _mm512_mul_epu32(x, y);
        return _mm512_add_epi64(_mm512_add_epi64(x, y), _mm512_add_epi64(z, z));
    }

    inline void swapHalves(__m512i &a0, __m512i &a1)
    {
        __m512i t0;
        __m512i t1;

        t0 = _mm512_shuffle_i64x2(a0, a1, _MM_SHUFFLE(1, 0, 1, 0));
        t1 = _mm512_shuffle_i64x2(a0, a1, _MM_SHUFFLE(3, 2, 3, 2));

        a0 = t0;
        a1 = t1;
    }

    inline void swapQuarters(__m512i &a0, __m512i &a1)
    {
        swapHalves(a0, a1);
        a0 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), a0);
        a1 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), a1);
    }

    inline void unswapQuarters(__m512i &a0, __m512i &a1)
    {
        a0 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), a0);
        a1 = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), a1);
        swapHalves(a0, a1);
    }

    void blamkaG1AVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1);

    void blamkaG2AVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1);

    void diagonalizeAVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1);

    void undiagonalizeAVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1);

    void Round(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1);

    void Round1(
        __m512i& a0, __m512i& c0, __m512i& b0, __m512i& d0,
        __m512i& a1, __m512i& c1, __m512i& b1, __m512i& d1);

    void Round2(
        __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
        __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1);

    void processBlockAVX512(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor);
}
