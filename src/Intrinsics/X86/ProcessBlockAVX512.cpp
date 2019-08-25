// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

/*
 * Argon2 reference source code package - reference C implementations
 *
 * Copyright 2015
 * Daniel Dinu, Dmitry Khovratovich, Jean-Philippe Aumasson, and Samuel Neves
 *
 * You may use this work under the terms of a Creative Commons CC0 1.0
 * License/Waiver or the Apache Public License 2.0, at your option. The terms of
 * these licenses can be found at:
 *
 * - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
 * - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0
 *
 * You should have received a copy of both of these licenses along with this
 * software. If not, they may be obtained at the above URLs.
 */

//////////////////////////////////////////////
#include "Intrinsics/X86/ProcessBlockAVX512.h"
//////////////////////////////////////////////

#include <cstring>

#include "Intrinsics/X86/RotationsAVX512.h"

namespace ProcessBlockAVX512
{
    void blamkaG1AVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1)
    {
        a0 = muladd(a0, b0);
        a1 = muladd(a1, b1);

        d0 = _mm512_xor_si512(d0, a0);
        d1 = _mm512_xor_si512(d1, a1);

        d0 = RotationsAVX512::rotr32(d0);
        d1 = RotationsAVX512::rotr32(d1);
        
        c0 = muladd(c0, d0);
        c1 = muladd(c1, d1);

        b0 = _mm512_xor_si512(b0, c0);
        b1 = _mm512_xor_si512(b1, c1);

        b0 = RotationsAVX512::rotr24(b0);
        b1 = RotationsAVX512::rotr24(b1);
    }

    void blamkaG2AVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1)
    {
        a0 = muladd(a0, b0);
        a1 = muladd(a1, b1);

        d0 = _mm512_xor_si512(d0, a0);
        d1 = _mm512_xor_si512(d1, a1);

        d0 = RotationsAVX512::rotr16(d0);
        d1 = RotationsAVX512::rotr16(d1);
        
        c0 = muladd(c0, d0);
        c1 = muladd(c1, d1);

        b0 = _mm512_xor_si512(b0, c0);
        b1 = _mm512_xor_si512(b1, c1);

        b0 = RotationsAVX512::rotr63(b0);
        b1 = RotationsAVX512::rotr63(b1);
    }

    void diagonalizeAVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1)
    {
        b0 = _mm512_permutex_epi64(b0, _MM_SHUFFLE(0, 3, 2, 1));
        b1 = _mm512_permutex_epi64(b1, _MM_SHUFFLE(0, 3, 2, 1));

        c0 = _mm512_permutex_epi64(c0, _MM_SHUFFLE(1, 0, 3, 2));
        c1 = _mm512_permutex_epi64(c1, _MM_SHUFFLE(1, 0, 3, 2));

        d0 = _mm512_permutex_epi64(d0, _MM_SHUFFLE(2, 1, 0, 3));
        d1 = _mm512_permutex_epi64(d1, _MM_SHUFFLE(2, 1, 0, 3));
    }

    void undiagonalizeAVX512(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1)
    {
        b0 = _mm512_permutex_epi64(b0, _MM_SHUFFLE(2, 1, 0, 3));
        b1 = _mm512_permutex_epi64(b1, _MM_SHUFFLE(2, 1, 0, 3));

        c0 = _mm512_permutex_epi64(c0, _MM_SHUFFLE(1, 0, 3, 2));
        c1 = _mm512_permutex_epi64(c1, _MM_SHUFFLE(1, 0, 3, 2));

        d0 = _mm512_permutex_epi64(d0, _MM_SHUFFLE(0, 3, 2, 1));
        d1 = _mm512_permutex_epi64(d1, _MM_SHUFFLE(0, 3, 2, 1));
    }

    void Round(
        __m512i& a0, __m512i& b0, __m512i& c0, __m512i& d0,
        __m512i& a1, __m512i& b1, __m512i& c1, __m512i& d1)
    {
        blamkaG1AVX512(a0, b0, c0, d0, a1, b1, c1, d1);
        blamkaG2AVX512(a0, b0, c0, d0, a1, b1, c1, d1);

        diagonalizeAVX512(a0, b0, c0, d0, a1, b1, c1, d1);

        blamkaG1AVX512(a0, b0, c0, d0, a1, b1, c1, d1);
        blamkaG2AVX512(a0, b0, c0, d0, a1, b1, c1, d1);

        undiagonalizeAVX512(a0, b0, c0, d0, a1, b1, c1, d1);
    }

    void Round1(
        __m512i& a0, __m512i& c0, __m512i& b0, __m512i& d0,
        __m512i& a1, __m512i& c1, __m512i& b1, __m512i& d1)
    {
        swapHalves(a0, b0);
        swapHalves(c0, d0);
        swapHalves(a1, b1);
        swapHalves(c1, d1);

        Round(a0, b0, c0, d0, a1, b1, c1, d1);

        swapHalves(a0, b0);
        swapHalves(c0, d0);
        swapHalves(a1, b1);
        swapHalves(c1, d1);
    }

    void Round2(
        __m512i& a0, __m512i& a1, __m512i& b0, __m512i& b1,
        __m512i& c0, __m512i& c1, __m512i& d0, __m512i& d1)
    {
        swapQuarters(a0, a1);
        swapQuarters(b0, b1);
        swapQuarters(c0, c1);
        swapQuarters(d0, d1);

        Round(a0, b0, c0, d0, a1, b1, c1, d1);

        unswapQuarters(a0, a1);
        unswapQuarters(b0, b1);
        unswapQuarters(c0, c1);
        unswapQuarters(d0, d1);
    }

    void processBlockAVX512(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor)
    {
        /* 16 * (512 / 8) = Constants::BLOCK_SIZE_BYTES */
        __m512i state[16];
        __m512i prevBlockIntrinsic[16];
        __m512i refBlockIntrinsic[16];

        /* Copy block */
        std::memcpy(state, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(refBlockIntrinsic, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(prevBlockIntrinsic, prevBlock.data(), Constants::BLOCK_SIZE_BYTES);

        /* Xor block */
        for (int i = 0; i < 16; i++)
        {
            state[i] = _mm512_xor_si512(state[i], prevBlockIntrinsic[i]);
        }

        for (uint32_t i = 0; i < 2; i++)
        {
            Round1(
                state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );
        }

        for (uint32_t i = 0; i < 2; i++)
        {
            Round2(
                state[2 * 0 + i], state[2 * 1 + i], state[2 * 2 + i], state[2 * 3 + i],
                state[2 * 4 + i], state[2 * 5 + i], state[2 * 6 + i], state[2 * 7 + i]
            );
        }

        if (doXor)
        {
            for (int i = 0; i < 16; i++)
            {
                /* nextBlock[i] = refBlock[i] ^ prevBlock[i] ^ state[i] */
                __m512i *blockToWrite = reinterpret_cast<__m512i *>(nextBlock.data()) + i;

                const auto _nextBlock =  _mm512_loadu_si512(blockToWrite);

                const __m512i stateXorPrev = _mm512_xor_si512(prevBlockIntrinsic[i], state[i]);
                const __m512i prevXorRef = _mm512_xor_si512(refBlockIntrinsic[i], stateXorPrev);
                const __m512i result = _mm512_xor_si512(_nextBlock, prevXorRef);

                _mm512_storeu_si512(blockToWrite, result);
            }
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                /* nextBlock[i] = refBlock[i] ^ prevBlock[i] ^ state[i] */
                __m512i *blockToWrite = reinterpret_cast<__m512i *>(nextBlock.data()) + i;

                const auto _nextBlock =  _mm512_loadu_si512(blockToWrite);

                const __m512i stateXorPrev = _mm512_xor_si512(prevBlockIntrinsic[i], state[i]);
                const __m512i result = _mm512_xor_si512(refBlockIntrinsic[i], stateXorPrev);

                _mm512_storeu_si512(blockToWrite, result);
            }
        }
    }
}
