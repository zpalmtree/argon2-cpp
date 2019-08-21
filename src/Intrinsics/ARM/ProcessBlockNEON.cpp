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

////////////////////////////////////////////
#include "Intrinsics/ARM/ProcessBlockNEON.h"
////////////////////////////////////////////

#include <cstring>

#include "Intrinsics/ARM/RotationsNEON.h"

namespace ProcessBlockNEON
{
    void blamkaG1NEON(
        uint64x2_t &a0, uint64x2_t &b0, uint64x2_t &c0, uint64x2_t &d0,
        uint64x2_t &a1, uint64x2_t &b1, uint64x2_t &c1, uint64x2_t &d1)
    {
        a0 = fblamka(a0, b0);
        a1 = fblamka(a1, b1);

        d0 = veorq_u64(d0, a0);
        d1 = veorq_u64(d1, a1);

        d0 = RotationsNEON::rotr32(d0);
        d1 = RotationsNEON::rotr32(d1);

        c0 = fblamka(c0, d0);
        c1 = fblamka(c1, d1);

        b0 = veorq_u64(b0, c0);
        b1 = veorq_u64(b1, c1);

        b0 = RotationsNEON::rotr24(b0);
        b1 = RotationsNEON::rotr24(b1);
    }

    void blamkaG2NEON(
        uint64x2_t &a0, uint64x2_t &b0, uint64x2_t &c0, uint64x2_t &d0,
        uint64x2_t &a1, uint64x2_t &b1, uint64x2_t &c1, uint64x2_t &d1)
    {
        a0 = fblamka(a0, b0);
        a1 = fblamka(a1, b1);

        d0 = veorq_u64(d0, a0);
        d1 = veorq_u64(d1, a1);

        d0 = RotationsNEON::rotr16(d0);
        d1 = RotationsNEON::rotr16(d1);

        c0 = fblamka(c0, d0);
        c1 = fblamka(c1, d1);

        b0 = veorq_u64(b0, c0);
        b1 = veorq_u64(b1, c1);

        b0 = RotationsNEON::rotr63(b0);
        b1 = RotationsNEON::rotr63(b1);
    }

    void diagonalizeNEON(
        uint64x2_t &a0, uint64x2_t &b0, uint64x2_t &c0, uint64x2_t &d0,
        uint64x2_t &a1, uint64x2_t &b1, uint64x2_t &c1, uint64x2_t &d1)
    {
        uint64x2_t t0 = vextq_u64(b0, b1, 1);
        uint64x2_t t1 = vextq_u64(b1, b0, 1);

        b0 = t0;
        b1 = t1;
        t0 = c0;
        c0 = c1;
        c1 = t0;

        t0 = vextq_u64(d1, d0, 1);
        t1 = vextq_u64(d0, d1, 1);

        d0 = t0;
        d1 = t1;
    }

    void undiagonalizeNEON(
        uint64x2_t &a0, uint64x2_t &b0, uint64x2_t &c0, uint64x2_t &d0,
        uint64x2_t &a1, uint64x2_t &b1, uint64x2_t &c1, uint64x2_t &d1)
    {
        uint64x2_t t0 = vextq_u64(b1, b0, 1);
        uint64x2_t t1 = vextq_u64(b0, b1, 1);

        b0 = t0;
        b1 = t1;
        t0 = c0;
        c0 = c1;
        c1 = t0;

        t0 = vextq_u64(d0, d1, 1);
        t1 = vextq_u64(d1, d0, 1);

        d0 = t0;
        d1 = t1;
    }

    void Round(
        uint64x2_t &a0, uint64x2_t &a1, uint64x2_t &b0, uint64x2_t &b1,
        uint64x2_t &c0, uint64x2_t &c1, uint64x2_t &d0, uint64x2_t &d1)
    {
	    blamkaG1NEON(a0, b0, c0, d0, a1, b1, c1, d1);
        blamkaG2NEON(a0, b0, c0, d0, a1, b1, c1, d1);

	    diagonalizeNEON(a0, b0, c0, d0, a1, b1, c1, d1);

        blamkaG1NEON(a0, b0, c0, d0, a1, b1, c1, d1);
        blamkaG2NEON(a0, b0, c0, d0, a1, b1, c1, d1);

        undiagonalizeNEON(a0, b0, c0, d0, a1, b1, c1, d1);
    }

    void processBlockNEON(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor)
    {
        /* 64 * (128 / 8) = Constants::BLOCK_SIZE_BYTES */
        uint64x2_t state[64];
        uint64x2_t prevBlockIntrinsic[64];
        uint64x2_t refBlockIntrinsic[64];

        /* Copy block */
        std::memcpy(state, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(refBlockIntrinsic, refBlock.data(), Constants::BLOCK_SIZE_BYTES);
        std::memcpy(prevBlockIntrinsic, prevBlock.data(), Constants::BLOCK_SIZE_BYTES);

        /* Xor block */
        for (int i = 0; i < 64; i++)
        {
            state[i] = veorq_u64(state[i], prevBlockIntrinsic[i]);
        }

        for (uint32_t i = 0; i < 8; i++)
        {
            Round(
                state[8 * i + 0], state[8 * i + 1], state[8 * i + 2], state[8 * i + 3],
                state[8 * i + 4], state[8 * i + 5], state[8 * i + 6], state[8 * i + 7]
            );
        }

        for (uint32_t i = 0; i < 8; i++)
        {
            Round(
                state[8 * 0 + i], state[8 * 1 + i], state[8 * 2 + i], state[8 * 3 + i],
                state[8 * 4 + i], state[8 * 5 + i], state[8 * 6 + i], state[8 * 7 + i]
            );
        }

        if (doXor)
        {
            for (int i = 0; i < 64; i++)
            {
                /* nextBlock[i] = refBlock[i] ^ prevBlock[i] ^ state[i] */
                uint64_t *blockToWrite = &nextBlock[i * 2];

                const auto _nextBlock =  vld1q_u64(blockToWrite);

                const uint64x2_t stateXorPrev = veorq_u64(prevBlockIntrinsic[i], state[i]);
                const uint64x2_t prevXorRef = veorq_u64(refBlockIntrinsic[i], stateXorPrev); 
                const uint64x2_t result = veorq_u64(_nextBlock, prevXorRef); 

                vst1q_u64(blockToWrite, result);
            }
        }
        else
        {
            for (int i = 0; i < 64; i++)
            {
                /* nextBlock[i] = refBlock[i] ^ prevBlock[i] ^ state[i] */
                uint64_t *blockToWrite = &nextBlock[i * 2];

                const auto _nextBlock =  vld1q_u64(blockToWrite);

                const uint64x2_t stateXorPrev = veorq_u64(prevBlockIntrinsic[i], state[i]);
                const uint64x2_t result = veorq_u64(refBlockIntrinsic[i], stateXorPrev); 

                vst1q_u64(blockToWrite, result);
            }
        }
    }
}
