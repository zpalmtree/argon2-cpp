// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include <cstddef>

#include "Argon2/Argon2.h"

void Argon2::processBlockGeneric(
    const Block &out,
    Block &in1,
    Block &in2,
    const bool doXor)
{
    processBlockGenericCrossPlatform(out, in1, in2, doXor);
}
