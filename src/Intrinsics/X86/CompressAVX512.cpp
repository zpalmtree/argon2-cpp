// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

//////////////////////////////////////////
#include "Intrinsics/X86/CompressAVX512.h"
//////////////////////////////////////////

#include "Intrinsics/X86/CompressAVX2.h"

namespace CompressAVX512
{
    void compressAVX512(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags)
    {
        return CompressAVX2::compressAVX2(hash, chunk, compressXorFlags);
    }
}
