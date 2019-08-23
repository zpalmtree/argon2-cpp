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

///////////////////////////////////////////
#include "Intrinsics/X86/ProcessBlockAVX2.h"
///////////////////////////////////////////

#include <cstring>

#include "Intrinsics/X86/RotationsAVX2.h"

namespace ProcessBlockAVX2
{
    void blamkaG1AVX2(
        __m256i& a0, __m256i& a1, __m256i& b0, __m256i& b1,
        __m256i& c0, __m256i& c1, __m256i& d0, __m256i& d1)
    {
        __m256i ml = _mm256_mul_epu32(a0, b0);
        ml = _mm256_add_epi64(ml, ml);
        a0 = _mm256_add_epi64(a0, _mm256_add_epi64(b0, ml));
        d0 = _mm256_xor_si256(d0, a0);
        d0 = RotationsAVX2::rotr32(d0);

        ml = _mm256_mul_epu32(c0, d0);
        ml = _mm256_add_epi64(ml, ml);
        c0 = _mm256_add_epi64(c0, _mm256_add_epi64(d0, ml));

        b0 = _mm256_xor_si256(b0, c0);
        b0 = RotationsAVX2::rotr24(b0);

        ml = _mm256_mul_epu32(a1, b1);
        ml = _mm256_add_epi64(ml, ml);
        a1 = _mm256_add_epi64(a1, _mm256_add_epi64(b1, ml));
        d1 = _mm256_xor_si256(d1, a1);
        d1 = RotationsAVX2::rotr32(d1);

        ml = _mm256_mul_epu32(c1, d1);
        ml = _mm256_add_epi64(ml, ml);
        c1 = _mm256_add_epi64(c1, _mm256_add_epi64(d1, ml));

        b1 = _mm256_xor_si256(b1, c1);
        b1 = RotationsAVX2::rotr24(b1);
    }

    void blamkaG2AVX2(
        __m256i& a0, __m256i& a1, __m256i& b0, __m256i& b1,
        __m256i& c0, __m256i& c1, __m256i& d0, __m256i& d1)
    {
        __m256i ml = _mm256_mul_epu32(a0, b0);
        ml = _mm256_add_epi64(ml, ml);
        a0 = _mm256_add_epi64(a0, _mm256_add_epi64(b0, ml));
        d0 = _mm256_xor_si256(d0, a0);
        d0 = RotationsAVX2::rotr16(d0);

        ml = _mm256_mul_epu32(c0, d0);
        ml = _mm256_add_epi64(ml, ml);
        c0 = _mm256_add_epi64(c0, _mm256_add_epi64(d0, ml));
        b0 = _mm256_xor_si256(b0, c0);
        b0 = RotationsAVX2::rotr63(b0);

        ml = _mm256_mul_epu32(a1, b1);
        ml = _mm256_add_epi64(ml, ml);
        a1 = _mm256_add_epi64(a1, _mm256_add_epi64(b1, ml));
        d1 = _mm256_xor_si256(d1, a1);
        d1 = RotationsAVX2::rotr16(d1);

        ml = _mm256_mul_epu32(c1, d1);
        ml = _mm256_add_epi64(ml, ml);
        c1 = _mm256_add_epi64(c1, _mm256_add_epi64(d1, ml));
        b1 = _mm256_xor_si256(b1, c1);
        b1 = RotationsAVX2::rotr63(b1);
    }

    void diagonalizeAVX2v1(__m256i& b0, __m256i& c0, __m256i& d0, __m256i& b1, __m256i& c1, __m256i& d1)
    {
        /* (v4, v5, v6, v7) -> (v5, v6, v7, v4) */
        b0 = _mm256_permute4x64_epi64(b0, _MM_SHUFFLE(0, 3, 2, 1));
        /* (v8, v9, v10, v11) -> (v10, v11, v8, v9) */
        c0 = _mm256_permute4x64_epi64(c0, _MM_SHUFFLE(1, 0, 3, 2));
        /* (v12, v13, v14, v15) -> (v15, v12, v13, v14) */
        d0 = _mm256_permute4x64_epi64(d0, _MM_SHUFFLE(2, 1, 0, 3));

        b1 = _mm256_permute4x64_epi64(b1, _MM_SHUFFLE(0, 3, 2, 1));
        c1 = _mm256_permute4x64_epi64(c1, _MM_SHUFFLE(1, 0, 3, 2));
        d1 = _mm256_permute4x64_epi64(d1, _MM_SHUFFLE(2, 1, 0, 3));
    }

    void diagonalizeAVX2v2(__m256i& b0, __m256i& b1, __m256i& c0, __m256i& c1, __m256i& d0, __m256i& d1)
    {
        /* (v4, v5, v6, v7) -> (v5, v6, v7, v4) */
        __m256i tmp1 = _mm256_blend_epi32(b0, b1, 0b11001100); /* v4v7 */
        __m256i tmp2 = _mm256_blend_epi32(b0, b1, 0b00110011); /* v6v5 */
        b1 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); /* v7v4 */
        b0 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); /* v5v6 */

        /* (v8, v9, v10, v11) -> (v10, v11, v8, v9) */
        tmp1 = c0;
        c0 = c1;
        c1 = tmp1;

        /* (v12, v13, v14, v15) -> (v15, v12, v13, v14) */
        tmp1 = _mm256_blend_epi32(d0, d1, 0b11001100); /* v12v15 */
        tmp2 = _mm256_blend_epi32(d0, d1, 0b00110011); /* v14v13 */
        d0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); /* v15v12 */
        d1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); /* v13v14 */
    }

    void undiagonalizeAVX2v1(__m256i& b0, __m256i& c0, __m256i& d0, __m256i& b1, __m256i& c1, __m256i& d1)
    {
        /* (v5, v6, v7, v4) -> (v4, v5, v6, v7) */
        b0 = _mm256_permute4x64_epi64(b0, _MM_SHUFFLE(2, 1, 0, 3));
        /* (v10, v11, v8, v9) -> (v8, v9, v10, v11) */
        c0 = _mm256_permute4x64_epi64(c0, _MM_SHUFFLE(1, 0, 3, 2));
        /* (v15, v12, v13, v14) -> (v12, v13, v14, v15) */
        d0 = _mm256_permute4x64_epi64(d0, _MM_SHUFFLE(0, 3, 2, 1));

        b1 = _mm256_permute4x64_epi64(b1, _MM_SHUFFLE(2, 1, 0, 3));
        c1 = _mm256_permute4x64_epi64(c1, _MM_SHUFFLE(1, 0, 3, 2));
        d1 = _mm256_permute4x64_epi64(d1, _MM_SHUFFLE(0, 3, 2, 1));
    }

    void undiagonalizeAVX2v2(__m256i& b0, __m256i& b1, __m256i& c0, __m256i& c1, __m256i& d0, __m256i& d1)
    {
        /* (v5, v6, v7, v4) -> (v4, v5, v6, v7) */
        __m256i tmp1 = _mm256_blend_epi32(b0, b1, 0b11001100); /* v5v4 */
        __m256i tmp2 = _mm256_blend_epi32(b0, b1, 0b00110011); /* v7v6 */
        b0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1)); /* v4v5 */
        b1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1)); /* v6v7 */

        /* (v10,v11,v8,v9) -> (v8,v9,v10,v11) */
        tmp1 = c0;
        c0 = c1;
        c1 = tmp1;

        /* (v15,v12,v13,v14) -> (v12,v13,v14,v15) */
        tmp1 = _mm256_blend_epi32(d0, d1, 0b00110011); /* v13v12 */
        tmp2 = _mm256_blend_epi32(d0, d1, 0b11001100); /* v15v14 */
        d0 = _mm256_permute4x64_epi64(tmp1, _MM_SHUFFLE(2,3,0,1));
        d1 = _mm256_permute4x64_epi64(tmp2, _MM_SHUFFLE(2,3,0,1));
    }

    void processBlockAVX2(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor)
    {
        /* 32 * (256 / 8) = Constants::BLOCK_SIZE_BYTES */
        __m256i state[32];
        __m256i prevBlockIntrinsic[32];
        __m256i refBlockIntrinsic[32];

        /* Copy block */
        std::memcpy(state, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(refBlockIntrinsic, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(prevBlockIntrinsic, prevBlock.data(), Constants::BLOCK_SIZE_BYTES);

        /* Xor block */
        for (int i = 0; i < 32; i++)
        {
            state[i] = _mm256_xor_si256(state[i], prevBlockIntrinsic[i]);
        }

        for (uint32_t i = 0; i < 4; i++)
        {
            blamkaG1AVX2(
                state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]
            );

            blamkaG2AVX2(
                state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]
            );

            diagonalizeAVX2v1(
                state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );

            blamkaG1AVX2(
                state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]
            );

            blamkaG2AVX2(
                state[8 * i + 0], state[8 * i + 4], state[8 * i + 1], state[8 * i + 5],
                state[8 * i + 2], state[8 * i + 6], state[8 * i + 3], state[8 * i + 7]
            );

            undiagonalizeAVX2v1(
                state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );
        }

        for(uint32_t i = 0; i < 4; i++)
        {
            blamkaG1AVX2(
                state[ 0 + i], state[ 4 + i], state[ 8 + i], state[12 + i],
                state[16 + i], state[20 + i], state[24 + i], state[28 + i]
            );

            blamkaG2AVX2(
                state[ 0 + i], state[ 4 + i], state[ 8 + i], state[12 + i],
                state[16 + i], state[20 + i], state[24 + i], state[28 + i]
            );

            diagonalizeAVX2v2(
                state[ 8 + i], state[12 + i],
                state[16 + i], state[20 + i],
                state[24 + i], state[28 + i]
            );

            blamkaG1AVX2(
                state[ 0 + i], state[ 4 + i], state[ 8 + i], state[12 + i],
                state[16 + i], state[20 + i], state[24 + i], state[28 + i]
            );

            blamkaG2AVX2(
                state[ 0 + i], state[ 4 + i], state[ 8 + i], state[12 + i],
                state[16 + i], state[20 + i], state[24 + i], state[28 + i]
            );

            undiagonalizeAVX2v2(
                state[ 8 + i], state[12 + i],
                state[16 + i], state[20 + i],
                state[24 + i], state[28 + i]
            );
        }
            
        if (doXor)
        {
            for (int i = 0; i < 32; i++)
            {
                /* nextBlock[i] ^= refBlock[i] ^ prevBlock[i] ^ state[i] */
                __m256i *blockToWrite = reinterpret_cast<__m256i *>(nextBlock.data()) + i;

                const auto _nextBlock =  _mm256_loadu_si256(blockToWrite);

                const __m256i stateXorPrev = _mm256_xor_si256(prevBlockIntrinsic[i], state[i]);
                const __m256i prevXorRef = _mm256_xor_si256(refBlockIntrinsic[i], stateXorPrev);
                const __m256i result = _mm256_xor_si256(_nextBlock, prevXorRef);

                _mm256_storeu_si256(blockToWrite, result);
            }
        }
        else
        {
            for (int i = 0; i < 32; i++)
            {
                /* nextBlock[i] = refBlock[i] ^ prevBlock[i] ^ state[i] */
                __m256i *blockToWrite = reinterpret_cast<__m256i *>(nextBlock.data()) + i;

                const auto _nextBlock =  _mm256_loadu_si256(blockToWrite);

                const __m256i stateXorPrev = _mm256_xor_si256(prevBlockIntrinsic[i], state[i]);
                const __m256i result = _mm256_xor_si256(refBlockIntrinsic[i], stateXorPrev);

                _mm256_storeu_si256(blockToWrite, result);
            }
        }
    }
}
