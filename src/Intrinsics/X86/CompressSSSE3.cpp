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

/////////////////////////////////////////
#include "Intrinsics/X86/CompressSSSE3.h"
/////////////////////////////////////////

#include "Blake2/Blake2b.h"
#include "Intrinsics/X86/BlakeIntrinsics.h"
#include "Intrinsics/X86/RotationsSSSE3.h"

namespace CompressSSSE3
{
    void g1SSSE3(
        __m128i& row1l, __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row1h, __m128i& row2h, __m128i& row3h, __m128i& row4h,
        __m128i& b0, __m128i& b1)
    {
        row1l = _mm_add_epi64(_mm_add_epi64(row1l, b0), row2l);
        row1h = _mm_add_epi64(_mm_add_epi64(row1h, b1), row2h);

        row4l = _mm_xor_si128(row4l, row1l);
        row4h = _mm_xor_si128(row4h, row1h);

        row4l = RotationsSSSE3::rotr32(row4l);
        row4h = RotationsSSSE3::rotr32(row4h);

        row3l = _mm_add_epi64(row3l, row4l);
        row3h = _mm_add_epi64(row3h, row4h);

        row2l = _mm_xor_si128(row2l, row3l);
        row2h = _mm_xor_si128(row2h, row3h);

        row2l = RotationsSSSE3::rotr24(row2l);
        row2h = RotationsSSSE3::rotr24(row2h);
    }

    void g2SSSE3(
        __m128i& row1l, __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row1h, __m128i& row2h, __m128i& row3h, __m128i& row4h,
        __m128i& b0, __m128i& b1)
    {
        row1l = _mm_add_epi64(_mm_add_epi64(row1l, b0), row2l);
        row1h = _mm_add_epi64(_mm_add_epi64(row1h, b1), row2h);

        row4l = _mm_xor_si128(row4l, row1l);
        row4h = _mm_xor_si128(row4h, row1h);

        row4l = RotationsSSSE3::rotr16(row4l);
        row4h = RotationsSSSE3::rotr16(row4h);

        row3l = _mm_add_epi64(row3l, row4l);
        row3h = _mm_add_epi64(row3h, row4h);

        row2l = _mm_xor_si128(row2l, row3l);
        row2h = _mm_xor_si128(row2h, row3h);

        row2l = RotationsSSSE3::rotr63(row2l);
        row2h = RotationsSSSE3::rotr63(row2h);
    }

    void diagonalizeSSSE3(
        __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row2h, __m128i& row3h, __m128i& row4h)
    {
        __m128i t0 = _mm_alignr_epi8(row2h, row2l, 8);
        __m128i t1 = _mm_alignr_epi8(row2l, row2h, 8);
        row2l = t0;
        row2h = t1;

        t0 = row3l;
        row3l = row3h;
        row3h = t0;

        t0 = _mm_alignr_epi8(row4h, row4l, 8);
        t1 = _mm_alignr_epi8(row4l, row4h, 8);
        row4l = t1;
        row4h = t0;
    }

    void undiagonalizeSSSE3(
        __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row2h, __m128i& row3h, __m128i& row4h)
    {
        __m128i t0 = _mm_alignr_epi8(row2l, row2h, 8);
        __m128i t1 = _mm_alignr_epi8(row2h, row2l, 8);
        row2l = t0;
        row2h = t1;

        t0 = row3l;
        row3l = row3h;
        row3h = t0;

        t0 = _mm_alignr_epi8(row4l, row4h, 8);
        t1 = _mm_alignr_epi8(row4h, row4l, 8);
        row4l = t1;
        row4h = t0;
    }

    void Round(
        int r,
        const uint64_t* block_ptr,
        __m128i& row1l, __m128i& row2l, __m128i& row3l, __m128i& row4l,
        __m128i& row1h, __m128i& row2h, __m128i& row3h, __m128i& row4h)
    {
        __m128i b0, b1;

        b0 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][2]], block_ptr[Blake2b::SIGMA[r][0]]);
        b1 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][6]], block_ptr[Blake2b::SIGMA[r][4]]);

        g1SSSE3(row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h, b0, b1);

        b0 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][3]], block_ptr[Blake2b::SIGMA[r][1]]);
        b1 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][7]], block_ptr[Blake2b::SIGMA[r][5]]);

        g2SSSE3(row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h, b0, b1);

        diagonalizeSSSE3(row2l,row3l,row4l,row2h,row3h,row4h);

        b0 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][10]], block_ptr[Blake2b::SIGMA[r][ 8]]);
        b1 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][14]], block_ptr[Blake2b::SIGMA[r][12]]);

        g1SSSE3(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h,b0,b1);

        b0 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][11]], block_ptr[Blake2b::SIGMA[r][ 9]]);
        b1 = _mm_set_epi64x(block_ptr[Blake2b::SIGMA[r][15]], block_ptr[Blake2b::SIGMA[r][13]]);

        g2SSSE3(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h,b0,b1);

        undiagonalizeSSSE3(row2l,row3l,row4l,row2h,row3h,row4h);
    }

    void compressSSSE3(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags)
    {
        __m128i row1l = _mm_loadu_si128(reinterpret_cast<__m128i *>(&hash[0]));
        __m128i row1h = _mm_loadu_si128(reinterpret_cast<__m128i *>(&hash[2]));
        __m128i row2l = _mm_loadu_si128(reinterpret_cast<__m128i *>(&hash[4]));
        __m128i row2h = _mm_loadu_si128(reinterpret_cast<__m128i *>(&hash[6]));

        __m128i row3l = IV128[0];
        __m128i row3h = IV128[1];

        __m128i row4l = _mm_xor_si128(IV128[2], _mm_loadu_si128(reinterpret_cast<__m128i *>(&compressXorFlags[0])));
        __m128i row4h = _mm_xor_si128(IV128[3], _mm_loadu_si128(reinterpret_cast<__m128i *>(&compressXorFlags[2])));

        for(int i = 0; i < 12; i++)
        {
            Round(i, chunk.data(), row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h);
        }

        _mm_storeu_si128(
            (__m128i*)&hash[0],
            _mm_xor_si128(_mm_loadu_si128((__m128i*)&hash[0]), _mm_xor_si128(row3l, row1l))
        );

        _mm_storeu_si128(
            (__m128i*)&hash[2],
            _mm_xor_si128(_mm_loadu_si128((__m128i*)&hash[2]), _mm_xor_si128(row3h, row1h))
        );

        _mm_storeu_si128(
            (__m128i*)&hash[4],
            _mm_xor_si128(_mm_loadu_si128((__m128i*)&hash[4]), _mm_xor_si128(row4l, row2l))
        );

        _mm_storeu_si128(
            (__m128i*)&hash[6],
            _mm_xor_si128(_mm_loadu_si128((__m128i*)&hash[6]), _mm_xor_si128(row4h, row2h))
        );
    }
}
