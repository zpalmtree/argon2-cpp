// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>

#include <vector>

typedef uint64_t Block[128];

std::vector<uint8_t> Argon2d(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t time, /* Or iterations */
    const uint32_t memory,
    const uint32_t threads, /* Or parallelism */
    const uint32_t keyLen /* Output hash length */);

std::vector<uint8_t> Argon2i(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t time, /* Or iterations */
    const uint32_t memory,
    const uint32_t threads, /* Or parallelism */
    const uint32_t keyLen /* Output hash length */);

std::vector<uint8_t> Argon2id(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t time, /* Or iterations */
    const uint32_t memory,
    const uint32_t threads, /* Or parallelism */
    const uint32_t keyLen /* Output hash length */);

std::vector<uint8_t> deriveKey(
    const uint32_t mode,
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const std::vector<uint8_t> &secret,
    const std::vector<uint8_t> &data,
    const uint32_t time,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t keyLen);

void validateParameters(
    const std::vector<uint8_t> &salt,
    const uint32_t threads,
    const uint32_t keyLen,
    const uint32_t memory,
    const uint32_t time,
    const uint32_t mode);

std::vector<uint8_t> initHash(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const std::vector<uint8_t> &key,
    const std::vector<uint8_t> &data,
    const uint32_t time,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t keyLen,
    const uint32_t mode);

std::vector<Block> initBlocks(
    std::vector<uint8_t> &h0,
    const uint32_t memory,
    const uint32_t threads);

void processBlocks(
    std::vector<Block> &B,
    const uint32_t time,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t mode);

void processSegment(
    const uint32_t n,
    const uint32_t slice,
    const uint32_t lane,
    std::vector<Block> &B,
    const uint32_t mode,
    const uint32_t memory,
    const uint32_t time,
    const uint32_t threads);

void blake2bHash(
    uint8_t *out,
    std::vector<uint8_t> input,
    uint32_t outputLength);

std::vector<uint8_t> extractKey(
    std::vector<Block> &block,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t keyLen);

void processBlockGeneric(
    Block &out,
    Block &in1,
    Block &in2,
    const bool doXor);

void blamkaGeneric(
    uint64_t &t00,
    uint64_t &t01,
    uint64_t &t02,
    uint64_t &t03,
    uint64_t &t04,
    uint64_t &t05,
    uint64_t &t06,
    uint64_t &t07,
    uint64_t &t08,
    uint64_t &t09,
    uint64_t &t10,
    uint64_t &t11,
    uint64_t &t12,
    uint64_t &t13,
    uint64_t &t14,
    uint64_t &t15);

void processBlock(
    Block &out,
    Block &in1,
    Block &in2);

void processBlockXOR(
    Block &out,
    Block &in1,
    Block &in2);

uint32_t indexAlpha(
    const uint64_t random,
    const uint32_t lanes,
    const uint32_t segments,
    const uint32_t threads,
    const uint32_t n,
    const uint32_t slice,
    const uint32_t lane,
    const uint32_t index);

uint32_t phi(
    const uint64_t random,
    uint64_t m,
    uint64_t s,
    const uint32_t lane,
    const uint32_t lanes);
