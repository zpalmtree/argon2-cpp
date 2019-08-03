// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include <cmath>

#include <cstddef>

namespace Constants
{
    constexpr uint32_t CURRENT_ARGON_VERSION = 19;

    enum ArgonVariant
    {
        ARGON2D = 0,
        ARGON2I = 1,
        ARGON2ID = 2,
    };

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
