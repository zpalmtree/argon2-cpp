// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

///////////////////////////////////////////
#include "Intrinsics/X86/BlakeIntrinsics.h"
///////////////////////////////////////////

#include "Intrinsics/X86/CompressAVX512.h"
#include "Intrinsics/X86/CompressAVX2.h"
#include "Intrinsics/X86/CompressSSE41.h"
#include "Intrinsics/X86/CompressSSSE3.h"
#include "Intrinsics/X86/CompressSSE2.h"

void Blake2b::compress()
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
        CompressAVX512::compressAVX512(m_hash, m_chunk, m_compressXorFlags);
    }
    else if (tryAVX2 && hasAVX2)
    {
        CompressAVX2::compressAVX2(m_hash, m_chunk, m_compressXorFlags);
    }
    else if (trySSE41 && hasSSE41)
    {
        CompressSSE41::compressSSE41(m_hash, m_chunk, m_compressXorFlags);
    }
    else if (trySSSE3 && hasSSSE3)
    {
        CompressSSSE3::compressSSSE3(m_hash, m_chunk, m_compressXorFlags);
    }
    else if (trySSE2 && hasSSE2)
    {
        CompressSSE2::compressSSE2(m_hash, m_chunk, m_compressXorFlags);
    }
    else
    {
        compressCrossPlatform();
    }
}
