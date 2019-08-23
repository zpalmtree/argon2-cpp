// Copyright (c) 2017, YANDEX LLC
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided 
// that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and 
// the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
// the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote 
// products derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

////////////////////////////////////////////
#include "Intrinsics/X86/ProcessBlockSSE2.h"
////////////////////////////////////////////

#include <cstring>

#include "Intrinsics/X86/RotationsSSE2.h"

namespace ProcessBlockSSE2
{
    void blamkaG1SSE2(
        __m128i& a0, __m128i& a1, __m128i& b0, __m128i& b1,
        __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1)
    {
        __m128i ml = _mm_mul_epu32(a0, b0);
        ml = _mm_add_epi64(ml, ml);
        a0 = _mm_add_epi64(a0, _mm_add_epi64(b0, ml));

        ml = _mm_mul_epu32(a1, b1);
        ml = _mm_add_epi64(ml, ml);
        a1 = _mm_add_epi64(a1, _mm_add_epi64(b1, ml));

        d0 = _mm_xor_si128(d0, a0);
        d1 = _mm_xor_si128(d1, a1);

        d0 = RotationsSSE2::rotr32(d0);
        d1 = RotationsSSE2::rotr32(d1);

        ml = _mm_mul_epu32(c0, d0);
        ml = _mm_add_epi64(ml, ml);
        c0 = _mm_add_epi64(c0, _mm_add_epi64(d0, ml));

        ml = _mm_mul_epu32(c1, d1);
        ml = _mm_add_epi64(ml, ml);
        c1 = _mm_add_epi64(c1, _mm_add_epi64(ml, d1));

        b0 = _mm_xor_si128(b0, c0);
        b1 = _mm_xor_si128(b1, c1);

        b0 = RotationsSSE2::rotr24(b0);
        b1 = RotationsSSE2::rotr24(b1);
    }

    void blamkaG2SSE2(
        __m128i& a0, __m128i& a1, __m128i& b0, __m128i& b1,
        __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1)
    {
        __m128i ml = _mm_mul_epu32(a0, b0);
        ml = _mm_add_epi64(ml, ml);
        a0 = _mm_add_epi64(a0, _mm_add_epi64(b0, ml));

        ml = _mm_mul_epu32(a1, b1);
        ml = _mm_add_epi64(ml, ml);
        a1 = _mm_add_epi64(a1, _mm_add_epi64(b1, ml));

        d0 = _mm_xor_si128(d0, a0);
        d1 = _mm_xor_si128(d1, a1);

        d0 = RotationsSSE2::rotr16(d0);
        d1 = RotationsSSE2::rotr16(d1);

        ml = _mm_mul_epu32(c0, d0);
        ml = _mm_add_epi64(ml, ml);
        c0 = _mm_add_epi64(c0, _mm_add_epi64(d0, ml));

        ml = _mm_mul_epu32(c1, d1);
        ml = _mm_add_epi64(ml, ml);
        c1 = _mm_add_epi64(c1, _mm_add_epi64(ml, d1));

        b0 = _mm_xor_si128(b0, c0);
        b1 = _mm_xor_si128(b1, c1);

        b0 = RotationsSSE2::rotr63(b0);
        b1 = RotationsSSE2::rotr63(b1);
    }

    void diagonalizeSSE2(
        __m128i& b0, __m128i& b1, __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1)
    {
        __m128i tmp0 = d0;
        __m128i tmp1 = b0;

        d0 = c0;
        c0 = c1;
        c1 = d0;

        d0 = _mm_unpackhi_epi64(d1, _mm_unpacklo_epi64(tmp0, tmp0));
        d1 = _mm_unpackhi_epi64(tmp0, _mm_unpacklo_epi64(d1, d1));
        b0 = _mm_unpackhi_epi64(b0, _mm_unpacklo_epi64(b1, b1));
        b1 = _mm_unpackhi_epi64(b1, _mm_unpacklo_epi64(tmp1, tmp1));
    }

    void undiagonalizeSSE2(
        __m128i& b0, __m128i& b1, __m128i& c0, __m128i& c1, __m128i& d0, __m128i& d1)
    {
        __m128i tmp0 = c0;

        c0 = c1;
        c1 = tmp0;
        tmp0 = b0;

        __m128i tmp1 = d0;

        b0 = _mm_unpackhi_epi64(b1, _mm_unpacklo_epi64(b0, b0));
        b1 = _mm_unpackhi_epi64(tmp0, _mm_unpacklo_epi64(b1, b1));
        d0 = _mm_unpackhi_epi64(d0, _mm_unpacklo_epi64(d1, d1));
        d1 = _mm_unpackhi_epi64(d1, _mm_unpacklo_epi64(tmp1, tmp1));
    }

    void processBlockSSE2(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor)
    {
        /* 64 * (128 / 8) = Constants::BLOCK_SIZE_BYTES */
        __m128i state[64];
        __m128i prevBlockIntrinsic[64];
        __m128i refBlockIntrinsic[64];

        /* Copy block */
        std::memcpy(state, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(refBlockIntrinsic, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(prevBlockIntrinsic, prevBlock.data(), Constants::BLOCK_SIZE_BYTES);

        /* Xor block */
        for (int i = 0; i < 64; i++)
        {
            state[i] = _mm_xor_si128(state[i], prevBlockIntrinsic[i]);
        }

        for (uint32_t i = 0; i < 8; i++)
        {
            blamkaG1SSE2(
                state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );

            blamkaG2SSE2(
                state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );

            diagonalizeSSE2(
                state[8 * i + 2], state[8 * i + 3], state[8 * i + 4],
                state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );

            blamkaG1SSE2(
                state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );

            blamkaG2SSE2(
                state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );

            undiagonalizeSSE2(
                state[8 * i + 2], state[8 * i + 3], state[8 * i + 4],
                state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );
        }

        for (uint32_t i = 0; i < 8; i++)
        {
            blamkaG1SSE2(
                state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i], state[8 * 6 + i], state[8 * 7 + i]
            );

            blamkaG2SSE2(
                state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i], state[8 * 6 + i], state[8 * 7 + i]
            );

            diagonalizeSSE2(
                state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i],
                state[8 * 6 + i], state[8 * 7 + i]
            );

            blamkaG1SSE2(
                state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i], state[8 * 6 + i], state[8 * 7 + i]
            );

            blamkaG2SSE2(
                state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i], state[8 * 6 + i], state[8 * 7 + i]
            );

            undiagonalizeSSE2(
                state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i],
                state[8 * 6 + i], state[8 * 7 + i]
            );
        }
            
        if (doXor)
        {
            for (int i = 0; i < 64; i++)
            {
                /* nextBlock[i] ^= refBlock[i] ^ prevBlock[i] ^ state[i] */
                __m128i *blockToWrite = reinterpret_cast<__m128i *>(nextBlock.data()) + i;

                const auto _nextBlock =  _mm_loadu_si128(blockToWrite);

                const __m128i stateXorPrev = _mm_xor_si128(prevBlockIntrinsic[i], state[i]);
                const __m128i prevXorRef = _mm_xor_si128(refBlockIntrinsic[i], stateXorPrev);
                const __m128i result = _mm_xor_si128(_nextBlock, prevXorRef);

                _mm_storeu_si128(blockToWrite, result);
            }
        }
        else
        {
            for (int i = 0; i < 64; i++)
            {
                /* nextBlock[i] ^= refBlock[i] ^ prevBlock[i] ^ state[i] */
                __m128i *blockToWrite = reinterpret_cast<__m128i *>(nextBlock.data()) + i;

                const auto _nextBlock =  _mm_loadu_si128(blockToWrite);

                const __m128i stateXorPrev = _mm_xor_si128(prevBlockIntrinsic[i], state[i]);
                const __m128i result = _mm_xor_si128(refBlockIntrinsic[i], stateXorPrev);

                _mm_storeu_si128(blockToWrite, result);
            }
        }
    }
}
