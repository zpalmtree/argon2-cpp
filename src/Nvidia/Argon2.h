// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <vector>

uint64_t *allocateScratchpads();

uint8_t *allocateResults();

void freeMemory(uint64_t *grids, uint8_t *results);

std::vector<uint8_t> nvidiaHash(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &saltInput,
    const uint32_t memory,
    const uint32_t iterations,
    const uint32_t gpuIndex,
    const uint32_t localNonce,
    uint64_t *grid,
    uint8_t *result);
