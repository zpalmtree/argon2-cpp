// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>

#include "Blake2/Blake2b.h"
#include "cpu_features/include/cpuinfo_x86.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

static const cpu_features::X86Features features = cpu_features::GetX86Info().features;

static const bool hasAVX512 = features.avx512f;
static const bool hasAVX2 = features.avx2;
static const bool hasSSE41 = features.sse4_1;
static const bool hasSSSE3 = features.ssse3;
static const bool hasSSE2 = features.sse2;

static const __m128i IV128[4] = {
    _mm_set_epi64x(0xbb67ae8584caa73bULL, 0x6a09e667f3bcc908ULL),
    _mm_set_epi64x(0xa54ff53a5f1d36f1ULL, 0x3c6ef372fe94f82bULL),
    _mm_set_epi64x(0x9b05688c2b3e6c1fULL, 0x510e527fade682d1ULL),
    _mm_set_epi64x(0x5be0cd19137e2179ULL, 0x1f83d9abfb41bd6bULL)
};
