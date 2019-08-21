/*
   BLAKE2 reference source code package - reference C implementations

   Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.  You may use this under the
   terms of the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at
   your option.  The terms of these licenses can be found at:

   - CC0 1.0 Universal : http://creativecommons.org/publicdomain/zero/1.0
   - OpenSSL license   : https://www.openssl.org/source/license.html
   - Apache 2.0        : http://www.apache.org/licenses/LICENSE-2.0

   More information about the BLAKE2 hash function can be found at
   https://blake2.net.
*/

// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

////////////////////////////////////////
#include "Intrinsics/ARM/CompressNEON.h"
////////////////////////////////////////

#include "Blake2/Blake2b.h"
#include "Intrinsics/ARM/BlakeIntrinsics.h"
#include "Intrinsics/ARM/RotationsNEON.h"
#include "Intrinsics/ARM/LoadNEON.h"

namespace CompressNEON
{
    void g1NEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h,
        uint64x2_t& b0, uint64x2_t& b1)
    {
        row1l = vaddq_u64(vaddq_u64(row1l, b0), row2l);
        row1h = vaddq_u64(vaddq_u64(row1h, b1), row2h);

        row4l = veorq_u64(row4l, row1l);
        row4h = veorq_u64(row4h, row1h);

        row4l = RotationsNEON::rotr32(row4l);
        row4h = RotationsNEON::rotr32(row4h);

        row3l = vaddq_u64(row3l, row4l);
        row3h = vaddq_u64(row3h, row4h);

        row2l = veorq_u64(row2l, row3l);
        row2h = veorq_u64(row2h, row3h);

        row2l = RotationsNEON::rotr24(row2l);
        row2h = RotationsNEON::rotr24(row2h);
    }

    void g2NEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h,
        uint64x2_t& b0, uint64x2_t& b1)
    {
        row1l = vaddq_u64(vaddq_u64(row1l, b0), row2l);
        row1h = vaddq_u64(vaddq_u64(row1h, b1), row2h);

        row4l = veorq_u64(row4l, row1l);
        row4h = veorq_u64(row4h, row1h);

        row4l = RotationsNEON::rotr16(row4l);
        row4h = RotationsNEON::rotr16(row4h);

        row3l = vaddq_u64(row3l, row4l);
        row3h = vaddq_u64(row3h, row4h);

        row2l = veorq_u64(row2l, row3l);
        row2h = veorq_u64(row2h, row3h);

        row2l = RotationsNEON::rotr63(row2l);
        row2h = RotationsNEON::rotr63(row2h);
    }

    void diagonalizeNEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h)
    {
        uint64x2_t t0 = vextq_u64(row2l, row2h, 1);
        uint64x2_t t1 = vextq_u64(row2h, row2l, 1);

        row2l = t0;
        row2h = t1;

        t0 = row3l;

        row3l = row3h;
        row3h = t0;

        t0 = vextq_u64(row4h, row4l, 1);
        t1 = vextq_u64(row4l, row4h, 1);

        row4l = t0;
        row4h = t1;
    }

    void undiagonalizeNEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h)
    {
        uint64x2_t t0 = vextq_u64(row2h, row2l, 1);
        uint64x2_t t1 = vextq_u64(row2l, row2h, 1);

        row2l = t0;
        row2h = t1;

        t0 = row3l;

        row3l = row3h;
        row3h = t0;

        t0 = vextq_u64(row4l, row4h, 1);
        t1 = vextq_u64(row4h, row4l, 1);

        row4l = t0;
        row4h = t1;
    }

    #define ROUND(r) \
        LOAD_MSG_ ##r ##_1(b0, b1); \
        g1NEON(row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h, b0, b1); \
        LOAD_MSG_ ##r ##_2(b0, b1); \
        g2NEON(row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h, b0, b1); \
        diagonalizeNEON(row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h); \
        LOAD_MSG_ ##r ##_3(b0, b1); \
        g1NEON(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h,b0,b1); \
        LOAD_MSG_ ##r ##_4(b0, b1); \
        g2NEON(row1l,row2l,row3l,row4l,row1h,row2h,row3h,row4h,b0,b1); \
        undiagonalizeNEON(row1l, row2l, row3l, row4l, row1h, row2h, row3h, row4h); \

    void compressNEON(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags)
    {
        /* These vars are used in LOAD_MSG */
        const uint64x2_t m0 = vld1q_u64(&chunk[0]);
        const uint64x2_t m1 = vld1q_u64(&chunk[2]);
        const uint64x2_t m2 = vld1q_u64(&chunk[4]);
        const uint64x2_t m3 = vld1q_u64(&chunk[6]);
        const uint64x2_t m4 = vld1q_u64(&chunk[8]);
        const uint64x2_t m5 = vld1q_u64(&chunk[10]);
        const uint64x2_t m6 = vld1q_u64(&chunk[12]);
        const uint64x2_t m7 = vld1q_u64(&chunk[14]);

        uint64x2_t row1l, row1h, row2l, row2h;
        uint64x2_t t0, t1, b0, b1;

        const uint64x2_t h0 = row1l = vld1q_u64(&hash[0]);
        const uint64x2_t h1 = row1h = vld1q_u64(&hash[2]);
        const uint64x2_t h2 = row2l = vld1q_u64(&hash[4]);
        const uint64x2_t h3 = row2h = vld1q_u64(&hash[6]);

        uint64x2_t row3l = vld1q_u64(&Blake2b::IV[0]);
        uint64x2_t row3h = vld1q_u64(&Blake2b::IV[2]);
        uint64x2_t row4l = veorq_u64(vld1q_u64(&Blake2b::IV[4]), vld1q_u64(&compressXorFlags[0]));
        uint64x2_t row4h = veorq_u64(vld1q_u64(&Blake2b::IV[6]), vld1q_u64(&compressXorFlags[2]));

        ROUND(0);
        ROUND(1);
        ROUND(2);
        ROUND(3);
        ROUND(4);
        ROUND(5);
        ROUND(6);
        ROUND(7);
        ROUND(8);
        ROUND(9);
        ROUND(10);
        ROUND(11);

        vst1q_u64(&hash[0], veorq_u64(h0, veorq_u64(row1l, row3l)));
        vst1q_u64(&hash[2], veorq_u64(h1, veorq_u64(row1h, row3h)));
        vst1q_u64(&hash[4], veorq_u64(h2, veorq_u64(row2l, row4l)));
        vst1q_u64(&hash[6], veorq_u64(h3, veorq_u64(row2h, row4h)));
    }
}
