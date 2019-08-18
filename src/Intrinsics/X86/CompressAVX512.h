// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>
#include <vector>

#include "Intrinsics/X86/IncludeIntrinsics.h"

namespace CompressAVX512
{
    void compressAVX512(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags);
}
