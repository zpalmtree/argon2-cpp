// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

///////////////////////////////////////////
#include "Intrinsics/ARM/ArgonIntrinsics.h"
///////////////////////////////////////////

#include <cstddef>
#include <cstring>

#include "Argon2/Argon2.h"
#include "Intrinsics/ARM/ProcessBlockNEON.h"
#include "Intrinsics/ARM/ProcessBlockARMv8.h"

void Argon2::processBlockGeneric(
    Block &nextBlock,
    const Block &refBlock,
    const Block &prevBlock,
    const bool doXor)
{
    /* NEON disabled by default unless explicitly specified.
       https://github.com/weidai11/cryptopp/issues/367 */
    if (m_optimizationMethod == Constants::NEON && hasNEON)
    {
        ProcessBlockNEON::processBlockNEON(nextBlock, refBlock, prevBlock, doXor);
    }
    else if (m_optimizationMethod == Constants::ARMV8 && hasARMV8)
    {
	if (doXor)
            ProcessBlockARMv8::processBlockARMv8DoXor(nextBlock, refBlock, prevBlock);
	else
            ProcessBlockARMv8::processBlockARMv8NoXor(nextBlock, refBlock, prevBlock);
    }
    else
    {
        processBlockGenericCrossPlatform(nextBlock, refBlock, prevBlock, doXor);
    }
}
