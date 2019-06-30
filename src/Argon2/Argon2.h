// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>

#include <vector>

std::vector<uint8_t> Argon2d(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations);

std::vector<uint8_t> Argon2i(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations);

std::vector<uint8_t> Argon2id(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations);

std::vector<uint8_t> Argon2Internal(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations,
    const uint32_t version,
    const std::vector<uint8_t> key,
    const std::vector<uint8_t> associatedData,
    const uint32_t hashType);
