// Copyright (c) 2021 notasailor
//
// Please see the included LICENSE file for more information.

#pragma once

#include "Argon2/Argon2.h"

namespace ProcessBlockARMv8
{
    void processBlockARMv8(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor);
    void blamkaGeneric(
    uint64_t &t00,
    uint64_t &t02,
    uint64_t &t04,
    uint64_t &t06,
    uint64_t &t08,
    uint64_t &t10,
    uint64_t &t12,
    uint64_t &t14);
}
