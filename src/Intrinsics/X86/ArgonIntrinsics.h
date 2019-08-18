// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>

#include "Argon2/Argon2.h"
#include "cpu_features/include/cpuinfo_x86.h"
#include "Intrinsics/X86/IncludeIntrinsics.h"

static const cpu_features::X86Features features = cpu_features::GetX86Info().features;

static const bool hasAVX512 = features.avx512f;
static const bool hasAVX2 = features.avx2;
static const bool hasSSE41 = features.sse4_1;
static const bool hasSSSE3 = features.ssse3;
static const bool hasSSE2 = features.sse2;

void processBlockSSE41(
    Block &nextBlock,
    const Block &refBlock,
    const Block &prevblock,
    const bool doXor);
