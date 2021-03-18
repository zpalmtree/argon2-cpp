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
}
