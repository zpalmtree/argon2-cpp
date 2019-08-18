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

#pragma once

#include <cstdint>

#include "Blake2/Blake2b.h"
#include "cpu_features/include/cpuinfo_x86.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

static const cpu_features::X86Features features = cpu_features::GetX86Info().features;

static const bool hasAVX512 = features.avx512f;
static const bool hasAVX2 = features.avx2;
static const bool hasSSE3 = features.sse3;
static const bool hasSSE2 = features.sse2;

__m256i rotr32(__m256i x);
__m256i rotr24(__m256i x);
__m256i rotr16(__m256i x);
__m256i rotr63(__m256i x);

void g1AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]);

void g2AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]);

void diagonalize(__m256i& b, __m256i& c, __m256i& d);

void undiagonalize(__m256i& b, __m256i& c, __m256i& d);

void compressAVX512(
    std::vector<uint64_t> &hash,
    std::vector<uint64_t> &chunk,
    std::vector<uint64_t> &compressXorFlags);

void compressAVX2(
    std::vector<uint64_t> &hash,
    std::vector<uint64_t> &chunk,
    std::vector<uint64_t> &compressXorFlags);

void compressSSE3(
    std::vector<uint64_t> &hash,
    std::vector<uint64_t> &chunk,
    std::vector<uint64_t> &compressXorFlags);

void compressSSE2(
    std::vector<uint64_t> &hash,
    std::vector<uint64_t> &chunk,
    std::vector<uint64_t> &compressXorFlags);
