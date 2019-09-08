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
    const uint32_t startNonce);

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
