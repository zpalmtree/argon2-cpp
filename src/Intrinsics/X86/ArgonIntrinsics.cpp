// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

///////////////////////////////////////////
#include "Intrinsics/X86/ArgonIntrinsics.h"
///////////////////////////////////////////

#include <cstddef>
#include <cstring>
#include <iostream>

#include "Argon2/Argon2.h"
#include "Intrinsics/X86/ProcessBlockAVX512.h"
#include "Intrinsics/X86/ProcessBlockAVX2.h"
#include "Intrinsics/X86/ProcessBlockSSSE3.h"
#include "Intrinsics/X86/ProcessBlockSSE2.h"

void processBlockSSE41(
    Block &nextBlock,
    const Block &refBlock,
    const Block &prevBlock,
    const bool doXor)
{
    /* SSE4.1 Implementation is the same as SSSE3 for blamka */
    return ProcessBlockSSSE3::processBlockSSSE3(nextBlock, refBlock, prevBlock, doXor);
}

void Argon2::processBlockGeneric(
    Block &nextBlock,
    const Block &refBlock,
    const Block &prevBlock,
    const bool doXor)
{
    const bool tryAVX512
        = m_optimizationMethod == Constants::AVX512 || m_optimizationMethod == Constants::AUTO;

    const bool tryAVX2
        = m_optimizationMethod == Constants::AVX2 || m_optimizationMethod == Constants::AUTO;

    const bool trySSE41
        = m_optimizationMethod == Constants::SSE41 || m_optimizationMethod == Constants::AUTO;

    const bool trySSSE3
        = m_optimizationMethod == Constants::SSSE3 || m_optimizationMethod == Constants::AUTO;

    const bool trySSE2
        = m_optimizationMethod == Constants::SSE2 || m_optimizationMethod == Constants::AUTO;

    if (tryAVX512 && hasAVX512)
    {
        ProcessBlockAVX512::processBlockAVX512(nextBlock, refBlock, prevBlock, doXor);
    }
    else if (tryAVX2 && hasAVX2)
    {
        ProcessBlockAVX2::processBlockAVX2(nextBlock, refBlock, prevBlock, doXor);
    }
    else if (trySSE41 && hasSSE41)
    {
        processBlockSSE41(nextBlock, refBlock, prevBlock, doXor);
    }
    else if (trySSSE3 && hasSSSE3)
    {
        ProcessBlockSSSE3::processBlockSSSE3(nextBlock, refBlock, prevBlock, doXor);
    }
    else if (trySSE2 && hasSSE2)
    {
        ProcessBlockSSE2::processBlockSSE2(nextBlock, refBlock, prevBlock, doXor);
    }
    else
    {
        processBlockGenericCrossPlatform(nextBlock, refBlock, prevBlock, doXor);
    }
}
