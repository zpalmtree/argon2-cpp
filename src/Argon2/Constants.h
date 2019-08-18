// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace Constants
{
    constexpr uint32_t CURRENT_ARGON_VERSION = 19;

    enum ArgonVariant : uint32_t
    {
        ARGON2D = 0,
        ARGON2I = 1,
        ARGON2ID = 2,
    };

    enum OptimizationMethod
    {
        AVX512 = 0,
        AVX2 = 1,
        SSE3 = 2,
        SSE2 = 3,
        NONE = 4,
        AUTO = 5,
    };

    inline std::string optimizationMethodToString(const OptimizationMethod method)
    {
        switch (method)
        {
            case AVX512:
            {
                return "AVX-512";
            }
            case AVX2:
            {
                return "AVX-2";
            }
            case SSE3:
            {
                return "SSE3";
            }
            case SSE2:
            {
                return "SSE2";
            }
            case NONE:
            {
                return "None";
            }
            case AUTO:
            {
                return "Auto";
            }
        }

        throw std::invalid_argument("Unknown optimization method!");
    }

    inline OptimizationMethod optimizationMethodFromString(const std::string &method)
    {
        if (method == "AVX-512")
        {
            return AVX512;
        }
        else if (method == "AVX-2")
        {
            return AVX2;
        }
        else if (method == "SSE3")
        {
            return SSE3;
        }
        else if (method == "SSE2")
        {
            return SSE2;
        }
        else if (method == "None")
        {
            return NONE;
        }
        else if (method == "Auto")
        {
            return AUTO;
        }

        throw std::invalid_argument("Optimization method " + method + " is unknown.");
    }

    /* Parallelism must be < 2^24 - 1 */
    constexpr uint32_t MAX_PARALLELISM = (1 << 24) - 1;

    /* Block size = 1KB = 128 * 64bit = 1024 bytes */
    constexpr uint32_t BLOCK_SIZE = 128;

    constexpr uint32_t BLOCK_SIZE_BYTES = BLOCK_SIZE * 8;

    /* Salt must be at least 8 bytes */
    constexpr uint8_t MIN_SALT_SIZE = 8;

    /* Output hash must be at least 4 bytes */
    constexpr uint8_t MIN_OUTPUT_HASH_LENGTH = 4;

    /* Memory usage must be at least 8 * parallelism */
    constexpr uint8_t MIN_PARALLELISM_FACTOR = 8;

    /* Split the blocks up into 4 lanes */
    constexpr uint8_t SYNC_POINTS = 4;

    /* Size of internal output hash function */
    constexpr uint8_t HASH_SIZE = 64;

    /* Size of initial hash with space for extra data */
    constexpr uint8_t INITIAL_HASH_SIZE = HASH_SIZE + 8;
}
