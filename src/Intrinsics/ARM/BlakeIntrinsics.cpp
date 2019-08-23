// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include "Blake2/Blake2b.h"
#include "Intrinsics/ARM/BlakeIntrinsics.h"
#include "Intrinsics/ARM/CompressNEON.h"

void Blake2b::compress()
{
    /* NEON disabled by default unless specifically specified.
       https://github.com/weidai11/cryptopp/issues/367 */
    if (m_optimizationMethod == Constants::NEON && hasNEON)
    {
        CompressNEON::compressNEON(m_hash, m_chunk, m_compressXorFlags);
    }
    else
    {
        compressCrossPlatform();
    }
}
