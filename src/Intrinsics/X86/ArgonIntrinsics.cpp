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
