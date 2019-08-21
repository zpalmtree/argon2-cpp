// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>
#include <vector>

#include <arm_neon.h>

namespace CompressNEON
{
    void g1NEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h,
        uint64x2_t& b0, uint64x2_t& b1);

    void g2NEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h,
        uint64x2_t& b0, uint64x2_t& b1);

    void diagonalizeNEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h);

    void undiagonalizeNEON(
        uint64x2_t& row1l, uint64x2_t& row2l, uint64x2_t& row3l, uint64x2_t& row4l,
        uint64x2_t& row1h, uint64x2_t& row2h, uint64x2_t& row3h, uint64x2_t& row4h);

    void compressNEON(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags);
}
