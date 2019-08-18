// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

// Copyright (c) 2017, YANDEX LLC
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided 
// that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and 
// the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
// the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote 
// products derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
