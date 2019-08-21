// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include "Blake2/Blake2b.h"
#include "Intrinsics/ARM/BlakeIntrinsics.h"
#include "Intrinsics/ARM/CompressNEON.h"

void Blake2b::compress()
{
    bool tryNEON = m_optimizationMethod == Constants::NEON;

    /* Only enable NEON by default on Armv7. https://github.com/weidai11/cryptopp/issues/367 */
    #if defined(ARMV7_OPTIMIZATIONS)
    tryNEON = tryNEON || m_optimizationMethod == Constants::AUTO;
    #endif

    if (tryNEON && hasNEON)
    {
        CompressNEON::compressNEON(m_hash, m_chunk, m_compressXorFlags);
    }
    else
    {
        compressCrossPlatform();
    }
}
