// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

namespace RotationsNEON
{
    inline uint64x2_t rotr32(uint64x2_t x)
    {
        return vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(x)));
    }

    inline uint64x2_t rotr24(uint64x2_t x)
    {
        return vcombine_u64(
            vreinterpret_u64_u8(vext_u8(vreinterpret_u8_u64(vget_low_u64(x)), vreinterpret_u8_u64(vget_low_u64(x)), 3)),
            vreinterpret_u64_u8(vext_u8(vreinterpret_u8_u64(vget_high_u64(x)), vreinterpret_u8_u64(vget_high_u64(x)), 3))
        );
    }

    inline uint64x2_t rotr16(uint64x2_t x)
    {
        return vcombine_u64(
            vreinterpret_u64_u8(vext_u8(vreinterpret_u8_u64(vget_low_u64(x)), vreinterpret_u8_u64(vget_low_u64(x)), 2)),
            vreinterpret_u64_u8(vext_u8(vreinterpret_u8_u64(vget_high_u64(x)), vreinterpret_u8_u64(vget_high_u64(x)), 2))
        );
    }

    inline uint64x2_t rotr63(uint64x2_t x)
    {
        return veorq_u64(vaddq_u64(x, x), vshrq_n_u64(x, 63));
    }
}
