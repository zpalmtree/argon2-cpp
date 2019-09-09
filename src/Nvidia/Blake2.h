// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <vector>

#include "Argon2.h"

__global__
void initMemoryKernel(
    struct block_g *memory,
    uint64_t *inseed,
    uint32_t memory_cost,
    uint32_t start_nonce,
    size_t blakeInputSize);

__global__
void getNonceKernel(
    block_g *memory,
    const uint32_t startNonce,
    uint64_t target,
    uint32_t *resultNonce,
    uint8_t *resultHash,
    bool *success);

void setupBlakeInput(
    const std::vector<uint8_t> &input,
    const std::vector<uint8_t> &saltInput,
    NvidiaState &state);
