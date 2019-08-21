// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include "Blake2/Blake2b.h"

#if defined(ARMV7_OPTIMIZATIONS)
#include "cpu_features/include/cpuinfo_arm.h"
static const cpu_features::ArmFeatures features = cpu_features::GetX86Info().features;
static const bool hasNEON = features.neon;
#else
static const bool hasNEON = true;
#endif
