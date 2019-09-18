// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <vector>

#include "Argon2.h"

__global__
void initMemoryKernel(
    block_g *memory,
    uint64_t *blakeInput,
    size_t blakeInputSize,
    const uint32_t startNonce,
    const size_t scratchpadSize,
    const uint64_t nonceMask);

__global__
void getNonceKernel(
    block_g *memory,
    const uint32_t startNonce,
    uint64_t target,
    uint32_t *resultNonce,
    uint8_t *resultHash,
    bool *success,
    const size_t scratchpadSize,
    const bool isNiceHash,
    const uint64_t *blakeInput);

void setupBlakeInput(
    const std::vector<uint8_t> &input,
    const std::vector<uint8_t> &saltInput,
    NvidiaState &state);
