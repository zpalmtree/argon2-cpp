// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <array>

#include <cstdint>

#include <vector>

#include "Argon2/Constants.h"

typedef std::array<uint64_t, 128> Block;

class Argon2
{
    public:
        /* CONSTRUCTOR */

        Argon2(
            const Constants::ArgonVariant mode,
            const std::vector<uint8_t> &secret,
            const std::vector<uint8_t> &data,
            const uint32_t time,
            const uint32_t memory,
            const uint32_t threads,
            const uint32_t keyLen,
            const Constants::OptimizationMethod optimizationMethod = Constants::AUTO);

        /* PUBLIC STATIC METHODS */

        static std::vector<uint8_t> Argon2d(
            const std::vector<uint8_t> &message,
            const std::vector<uint8_t> &salt,
            const uint32_t time, /* Or iterations */
            const uint32_t memory,
            const uint32_t threads, /* Or parallelism */
            const uint32_t keyLen /* Output hash length */);

        static std::vector<uint8_t> Argon2i(
            const std::vector<uint8_t> &message,
            const std::vector<uint8_t> &salt,
            const uint32_t time, /* Or iterations */
            const uint32_t memory,
            const uint32_t threads, /* Or parallelism */
            const uint32_t keyLen /* Output hash length */);

        static std::vector<uint8_t> Argon2id(
            const std::vector<uint8_t> &message,
            const std::vector<uint8_t> &salt,
            const uint32_t time, /* Or iterations */
            const uint32_t memory,
            const uint32_t threads, /* Or parallelism */
            const uint32_t keyLen /* Output hash length */);

        static std::vector<uint8_t> DeriveKey(
            const Constants::ArgonVariant mode,
            const std::vector<uint8_t> &message,
            const std::vector<uint8_t> &salt,
            const std::vector<uint8_t> &secret,
            const std::vector<uint8_t> &data,
            const uint32_t time,
            const uint32_t memory,
            const uint32_t threads,
            const uint32_t keyLen);

        /* PUBLIC METHODS */

        std::vector<uint8_t> Hash(
            const std::vector<uint8_t> &message,
            const std::vector<uint8_t> &salt);

    private:
        /* DEFINITIONS */

        //typedef uint64_t Block[128];

        /* PRIVATE METHODS */

        void validateParameters();

        std::vector<uint8_t> initHash(
            const std::vector<uint8_t> &message,
            const std::vector<uint8_t> &salt);

        void initBlocks(std::vector<uint8_t> &h0);

        void processBlocks();

        void processSegment(
            const uint32_t n,
            const uint32_t slice,
            const uint32_t lane);

        std::vector<uint8_t> extractKey();

        void blake2bHash(
            uint8_t *out,
            std::vector<uint8_t> input,
            uint32_t outputLength);

        void processBlockGeneric(
            Block &out,
            const Block &in1,
            const Block &in2,
            const bool doXor);

        void processBlockGenericCrossPlatform(
            Block &out,
            const Block &in1,
            const Block &in2,
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
            const Block &in1,
            const Block &in2);

        void processBlockXOR(
            Block &out,
            const Block &in1,
            const Block &in2);

        uint32_t indexAlpha(
            const uint64_t random,
            const uint32_t n,
            const uint32_t slice,
            const uint32_t lane,
            const uint32_t index);

        uint32_t phi(
            const uint64_t random,
            uint64_t m,
            uint64_t s,
            const uint32_t lane);

        /* PRIVATE VARIABLES */

        /* The argon variant to use */
        const Constants::ArgonVariant m_mode;

        /* The associated secret data */
        const std::vector<uint8_t> m_secret;

        /* The associated data */
        const std::vector<uint8_t> m_data;

        /* The time cost / iterations */
        const uint32_t m_time;

        /* The amount of memory to use, in KB */
        const uint32_t m_memory;

        /* The adjusted amount of memory to use, in KB */
        uint32_t m_scratchpadSize;

        /* The threads / parallism value */
        const uint32_t m_threads;

        /* The output hash length */
        const uint32_t m_keyLen;

        /* The argon version we are using */
        const uint32_t m_version = Constants::CURRENT_ARGON_VERSION;

        /* The scratchpad */
        std::vector<Block> m_B;

        /* Number of lanes to use */
        uint32_t m_lanes;

        /* Number of segments to use */
        uint32_t m_segments;

        /* Preferred optimization method to use */
        const Constants::OptimizationMethod m_optimizationMethod;
};
