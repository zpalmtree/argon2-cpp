// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include "Blake2/Blake2b.h"
#include "Intrinsics/ARM/BlakeIntrinsics.h"
#include "Intrinsics/ARM/CompressNEON.h"

void Blake2b::compress()
{
    const bool tryNEON = 
        m_optimizationMethod == Constants::NEON || m_optimizationMethod == Constants::AUTO;

    if (tryNEON && hasNEON)
    {
        CompressNEON::compressNEON(m_hash, m_chunk, m_compressXorFlags);
    }
    else
    {
        compressCrossPlatform();
    }
}
