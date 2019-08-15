// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>

#include "Blake2/Blake2b.h"
#include "cpu_features/include/cpuinfo_x86.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

static const cpu_features::X86Features features = cpu_features::GetX86Info().features;
static const bool hasAVX2 = features.avx2;

__m256i rotr32(__m256i x);
__m256i rotr24(__m256i x);
__m256i rotr16(__m256i x);
__m256i rotr63(__m256i x);

void g1AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]);

void g2AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]);

void diagonalize(__m256i& b, __m256i& c, __m256i& d);

void undiagonalize(__m256i& b, __m256i& c, __m256i& d);
