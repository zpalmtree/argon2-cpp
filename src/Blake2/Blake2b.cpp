// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

////////////////////
#include "Blake2b.h"
////////////////////

#include <array>

#include <cstring>

#include <stdexcept>

std::vector<uint8_t> Blake2b_Internal(
    std::vector<uint8_t> message,
    std::vector<uint8_t> key,
    const uint8_t outputHashLength);

void compress(
    std::vector<uint64_t> &hash,
    const std::vector<uint64_t> chunk,
    const uint64_t bytesCompressed,
    const bool finalChunk);

void mix(
    uint64_t &vA,
    uint64_t &vB,
    uint64_t &vC,
    uint64_t &vD,
    const uint64_t x,
    const uint64_t y);

/* https://stackoverflow.com/a/13732181/8737306 */
template<typename T>
T rotateRight(T x, unsigned int moves)
{
    return (x >> moves) | (x << sizeof(T) * 8 - moves);
}

/* Initialization vector */
constexpr std::array<uint64_t, 8> IV
{
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
};

/* Sigma round constants */
constexpr std::array<
    std::array<uint8_t, 16>,
    10
> SIGMA
{{
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 },
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  },
    { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  },
    { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  },
    { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 },
    { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  },
    { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 },
    { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 },
    { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  },
    { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  }
}};

std::vector<uint8_t> Blake2b(const std::vector<uint8_t> &message)
{
    return Blake2b_Internal(message, {}, 64);
}

std::vector<uint8_t> Blake2b(const std::string &message)
{
    return Blake2b_Internal({message.begin(), message.end()}, {}, 64);
}

std::vector<uint8_t> Blake2b_Internal(
    std::vector<uint8_t> message,
    std::vector<uint8_t> key,
    const uint8_t outputHashLength)
{
    if (outputHashLength > 64 || outputHashLength < 1)
    {
        throw std::invalid_argument("Invalid argument for outputHashLength. Must be between 1 and 64.");
    }

    if (key.size() > 64)
    {
        throw std::invalid_argument("Optional key must be at most 64 bytes");
    }

    std::vector<uint64_t> hash(8);

    /* Copy the IV to the hash */
    std::copy(IV.begin(), IV.end(), hash.begin());

    /* Mix key size and desired hash length into hash[0] */
    hash[0] ^= 0x01010000 ^ (key.size() << 8) ^ outputHashLength;

    size_t bytesCompressed = 0;
    size_t bytesRemaining = message.size();

    if (!key.empty())
    {
        /* If we were given a key, pad it with zeros to make it 128 bytes */
        key.resize(128, 0);

        /* Then prepend the key to the front of the message */
        message.insert(message.begin(), key.begin(), key.end());

        /* And now of course we have 128 more bytes to process. */
        bytesRemaining += 128;
    }

    /* 128 byte chunk to process */
    std::vector<uint64_t> chunk(16);

    /* Process 128 bytes at once, aside from final chunk */
    while (bytesRemaining > 128)
    {
        /* Copy 128 bytes of the message to the new chunk */
        std::memcpy(&chunk[0], &message[bytesCompressed], 128);

        /* Update the progress tally */
        bytesCompressed += 128;
        bytesRemaining -= 128;

        /* Perform the compression function (false = not final chunk) */
        compress(hash, chunk, bytesCompressed, false);
    }

    /* Pad the final chunk with zeros if needed */
    std::vector<uint64_t> finalChunk(16, 0);

    /* Copy the last chunk (may be less than 128 bytes) to finalChunk */
    std::memcpy(&finalChunk[0], &message[bytesCompressed], bytesRemaining);

    /* Update the progress tally */
    bytesCompressed += bytesRemaining;

    /* Perform the final compression (true = final chunk) */
    compress(hash, finalChunk, bytesCompressed, true);

    /* Return the final hash as a byte array */
    std::vector<uint8_t> finalHash(64);

    std::memcpy(&finalHash[0], &hash[0], 64);

    return finalHash;
}

void compress(
    std::vector<uint64_t> &hash,
    const std::vector<uint64_t> chunk,
    const uint64_t bytesCompressed,
    const bool finalChunk)
{
    std::vector<uint64_t> v(16);

    /* v[0..7] = h[0..7] */
    std::copy(hash.begin(), hash.end(), v.begin());

    /* v[8..15] = IV[0..7] */
    std::copy(IV.begin(), IV.end(), v.begin() + 8);

    /* Normally, this would be
        v[12] ^= LO(bytesCompressed)
        v[13] ^= HI(bytesCompressed),
      but we are not supporting messages > 2^64, so the high bits will always
      be zero. Since XOR with 0 is a no-op, we can skip the second operation. */
    v[12] ^= bytesCompressed;

    /* If this is the last block, then invert all the bits in v[14] */
    if (finalChunk)
    {
        v[14] ^= 0xFFFFFFFFFFFFFFFF;
    }

    /* 12 rounds of mixing */
    for (int i = 0; i < 12; i++)
    {
        /* Get the sigma constant for the current round */
        const auto &sigma = SIGMA[i % 10];

        mix(v[0], v[4], v[8],  v[12], chunk[sigma[0]],  chunk[sigma[1]]);
        mix(v[1], v[5], v[9],  v[13], chunk[sigma[2]],  chunk[sigma[3]]);
        mix(v[2], v[6], v[10], v[14], chunk[sigma[4]],  chunk[sigma[5]]);
        mix(v[3], v[7], v[11], v[15], chunk[sigma[6]],  chunk[sigma[7]]);

        mix(v[0], v[5], v[10], v[15], chunk[sigma[8]],  chunk[sigma[9]]);
        mix(v[1], v[6], v[11], v[12], chunk[sigma[10]], chunk[sigma[11]]);
        mix(v[2], v[7], v[8],  v[13], chunk[sigma[12]], chunk[sigma[13]]);
        mix(v[3], v[4], v[9],  v[14], chunk[sigma[14]], chunk[sigma[15]]);
    }

    for (int i = 0; i < 8; i++)
    {
        hash[i] ^= v[i] ^ v[i + 8];
    }
}

void mix(
    uint64_t &vA,
    uint64_t &vB,
    uint64_t &vC,
    uint64_t &vD,
    const uint64_t x,
    const uint64_t y)
{
    vA += vB + x;
    vD = rotateRight(vD ^ vA, 32);

    vC += vD;
    vB = rotateRight(vB ^ vC, 24);

    vA += vB + y;
    vD = rotateRight(vD ^ vA, 16);

    vC += vD;
    vB = rotateRight(vB ^ vC, 63);
}
