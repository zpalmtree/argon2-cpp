// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include <cstring>
#include <stdint.h>
#include <iostream>
#include <vector>

/* Sigma round constants */
__device__ __constant__ uint8_t SIGMA[12][16] = 
{
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 },
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  },
    { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  },
    { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  },
    { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 },
    { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  },
    { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 },
    { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 },
    { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  },
    { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  },
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 },
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  },
};

/* https://stackoverflow.com/a/13732181/8737306 */
template<typename T>
__device__ __forceinline__ T rotateRight(T x, unsigned int moves)
{
    return (x >> moves) | (x << (sizeof(T) * 8 - moves));
}

__device__ __forceinline__
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

__device__
void compress(uint64_t hash[8], uint64_t compressXorFlags[4], uint64_t chunk[16])
{
    /* Init hash with IV */
    uint64_t v[16] = 
    {
        hash[0],
        hash[1],
        hash[2],
        hash[3],
        hash[4],
        hash[5],
        hash[6],
        hash[7],
        0x6A09E667F3BCC908,
        0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B,
        0xA54FF53A5F1D36F1,
        0x510E527FADE682D1,
        0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B,
        0x5BE0CD19137E2179,
    };

    v[12] ^= compressXorFlags[0];
    v[13] ^= compressXorFlags[1];
    v[14] ^= compressXorFlags[2];
    v[15] ^= compressXorFlags[3];

    for (int i = 0; i < 12; i++)
    {
        const auto &sigma = SIGMA[i];

        /* Column round */
        mix(v[0], v[4], v[8],  v[12], chunk[sigma[0]],  chunk[sigma[1]]);
        mix(v[1], v[5], v[9],  v[13], chunk[sigma[2]],  chunk[sigma[3]]);
        mix(v[2], v[6], v[10], v[14], chunk[sigma[4]],  chunk[sigma[5]]);
        mix(v[3], v[7], v[11], v[15], chunk[sigma[6]],  chunk[sigma[7]]);

        /* Diagonal round */
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

__device__
void blake2bGPU(
    uint8_t *result,
    uint8_t *input,
    size_t inputLength,
    uint8_t outputHashLength) /* Note: 1 to 64 bytes */
{
    uint64_t compressXorFlags[4] = {};

    /* Init hash with IV */
    uint64_t hash[8] = 
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

    hash[0] ^= 0x01010000 ^ outputHashLength;

    uint64_t chunk[16] = {};

    uint8_t chunkSize = 0;

    size_t offset = 0;

    void *ptr = static_cast<void *>(&chunk[0]);

    while (inputLength > 0)
    {
        if (chunkSize == 128)
        {
            compress(hash, compressXorFlags, chunk);
            chunkSize = 0;
        }

        uint8_t size = 128 - chunkSize;

        if (size > inputLength)
        {
            size = static_cast<uint8_t>(inputLength);
        }

        ptr = static_cast<uint8_t *>(ptr) + chunkSize;

        std::memcpy(ptr, input + offset, size);

        chunkSize += size;

        /* compressXorFlags[0..1] is a 128 bit number stored in little endian. */
        /* Increase the bottom bits */
        compressXorFlags[0] += size;

        /* If it's less than the value we just added, we overflowed, and need to
           add one to the top bits */
        compressXorFlags[1] += (compressXorFlags[0] < size) ? 1 : 0;

        inputLength -= size;

        offset += size;
    }

    ptr = static_cast<void *>(&chunk[0]);
    ptr = static_cast<uint8_t *>(ptr) + chunkSize;

    /* Pad final chunk with zeros */
    std::memset(ptr, 0, 128 - chunkSize);

    /* Set all bytes, indicates last block */
    compressXorFlags[2] = 0xFFFFFFFFFFFFFFFF;

    /* Process final chunk */
    compress(hash, compressXorFlags, chunk);

    std::memcpy(result, &hash[0], outputHashLength);
}

const size_t BLOCK_SIZE = 128;
const size_t BLOCK_SIZE_BYTES = 128 * 8;
const size_t SYNC_POINTS = 4;
const size_t HASH_SIZE = 64;
const size_t INITIAL_HASH_SIZE = 72;
const size_t RESULT_HASH_SIZE = 32;

const uint32_t TRTL_MEMORY = 512;
const uint32_t TRTL_SCRATCHPAD_SIZE = TRTL_MEMORY;
const uint32_t TRTL_LANES = TRTL_SCRATCHPAD_SIZE;
const uint32_t TRTL_SALT_LENGTH = 16;
const uint32_t TRTL_ITERATIONS = 3;
const uint32_t TRTL_SEGMENTS = TRTL_LANES / SYNC_POINTS;

__device__ __forceinline__
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
    uint64_t &t15) {

    uint64_t v00 = t00, v01 = t01, v02 = t02, v03 = t03;
    uint64_t v04 = t04, v05 = t05, v06 = t06, v07 = t07;
    uint64_t v08 = t08, v09 = t09, v10 = t10, v11 = t11;
    uint64_t v12 = t12, v13 = t13, v14 = t14, v15 = t15;

    v00 += v04 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v00))*static_cast<uint64_t>(static_cast<uint32_t>(v04));
    v12 ^= v00;
    v12 = v12>>32 | v12<<32;
    v08 += v12 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v08))*static_cast<uint64_t>(static_cast<uint32_t>(v12));
    v04 ^= v08;
    v04 = v04>>24 | v04<<40;

    v00 += v04 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v00))*static_cast<uint64_t>(static_cast<uint32_t>(v04));
    v12 ^= v00;
    v12 = v12>>16 | v12<<48;
    v08 += v12 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v08))*static_cast<uint64_t>(static_cast<uint32_t>(v12));
    v04 ^= v08;
    v04 = v04>>63 | v04<<1;

    v01 += v05 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v01))*static_cast<uint64_t>(static_cast<uint32_t>(v05));
    v13 ^= v01;
    v13 = v13>>32 | v13<<32;
    v09 += v13 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v09))*static_cast<uint64_t>(static_cast<uint32_t>(v13));
    v05 ^= v09;
    v05 = v05>>24 | v05<<40;

    v01 += v05 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v01))*static_cast<uint64_t>(static_cast<uint32_t>(v05));
    v13 ^= v01;
    v13 = v13>>16 | v13<<48;
    v09 += v13 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v09))*static_cast<uint64_t>(static_cast<uint32_t>(v13));
    v05 ^= v09;
    v05 = v05>>63 | v05<<1;

    v02 += v06 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v02))*static_cast<uint64_t>(static_cast<uint32_t>(v06));
    v14 ^= v02;
    v14 = v14>>32 | v14<<32;
    v10 += v14 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v10))*static_cast<uint64_t>(static_cast<uint32_t>(v14));
    v06 ^= v10;
    v06 = v06>>24 | v06<<40;

    v02 += v06 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v02))*static_cast<uint64_t>(static_cast<uint32_t>(v06));
    v14 ^= v02;
    v14 = v14>>16 | v14<<48;
    v10 += v14 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v10))*static_cast<uint64_t>(static_cast<uint32_t>(v14));
    v06 ^= v10;
    v06 = v06>>63 | v06<<1;

    v03 += v07 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v03))*static_cast<uint64_t>(static_cast<uint32_t>(v07));
    v15 ^= v03;
    v15 = v15>>32 | v15<<32;
    v11 += v15 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v11))*static_cast<uint64_t>(static_cast<uint32_t>(v15));
    v07 ^= v11;
    v07 = v07>>24 | v07<<40;

    v03 += v07 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v03))*static_cast<uint64_t>(static_cast<uint32_t>(v07));
    v15 ^= v03;
    v15 = v15>>16 | v15<<48;
    v11 += v15 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v11))*static_cast<uint64_t>(static_cast<uint32_t>(v15));
    v07 ^= v11;
    v07 = v07>>63 | v07<<1;

    v00 += v05 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v00))*static_cast<uint64_t>(static_cast<uint32_t>(v05));
    v15 ^= v00;
    v15 = v15>>32 | v15<<32;
    v10 += v15 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v10))*static_cast<uint64_t>(static_cast<uint32_t>(v15));
    v05 ^= v10;
    v05 = v05>>24 | v05<<40;

    v00 += v05 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v00))*static_cast<uint64_t>(static_cast<uint32_t>(v05));
    v15 ^= v00;
    v15 = v15>>16 | v15<<48;
    v10 += v15 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v10))*static_cast<uint64_t>(static_cast<uint32_t>(v15));
    v05 ^= v10;
    v05 = v05>>63 | v05<<1;

    v01 += v06 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v01))*static_cast<uint64_t>(static_cast<uint32_t>(v06));
    v12 ^= v01;
    v12 = v12>>32 | v12<<32;
    v11 += v12 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v11))*static_cast<uint64_t>(static_cast<uint32_t>(v12));
    v06 ^= v11;
    v06 = v06>>24 | v06<<40;

    v01 += v06 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v01))*static_cast<uint64_t>(static_cast<uint32_t>(v06));
    v12 ^= v01;
    v12 = v12>>16 | v12<<48;
    v11 += v12 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v11))*static_cast<uint64_t>(static_cast<uint32_t>(v12));
    v06 ^= v11;
    v06 = v06>>63 | v06<<1;

    v02 += v07 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v02))*static_cast<uint64_t>(static_cast<uint32_t>(v07));
    v13 ^= v02;
    v13 = v13>>32 | v13<<32;
    v08 += v13 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v08))*static_cast<uint64_t>(static_cast<uint32_t>(v13));
    v07 ^= v08;
    v07 = v07>>24 | v07<<40;

    v02 += v07 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v02))*static_cast<uint64_t>(static_cast<uint32_t>(v07));
    v13 ^= v02;
    v13 = v13>>16 | v13<<48;
    v08 += v13 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v08))*static_cast<uint64_t>(static_cast<uint32_t>(v13));
    v07 ^= v08;
    v07 = v07>>63 | v07<<1;

    v03 += v04 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v03))*static_cast<uint64_t>(static_cast<uint32_t>(v04));
    v14 ^= v03;
    v14 = v14>>32 | v14<<32;
    v09 += v14 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v09))*static_cast<uint64_t>(static_cast<uint32_t>(v14));
    v04 ^= v09;
    v04 = v04>>24 | v04<<40;

    v03 += v04 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v03))*static_cast<uint64_t>(static_cast<uint32_t>(v04));
    v14 ^= v03;
    v14 = v14>>16 | v14<<48;
    v09 += v14 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v09))*static_cast<uint64_t>(static_cast<uint32_t>(v14));
    v04 ^= v09;
    v04 = v04>>63 | v04<<1;

    t00 = v00;
    t01 = v01;
    t02 = v02;
    t03 = v03;
    t04 = v04;
    t05 = v05;
    t06 = v06;
    t07 = v07;
    t08 = v08;
    t09 = v09;
    t10 = v10;
    t11 = v11;
    t12 = v12;
    t13 = v13;
    t14 = v14;
    t15 = v15;
}

__device__ __forceinline__
void processBlock(
    uint64_t *nextBlock,
    uint64_t *refBlock,
    uint64_t *prevBlock,
    const bool doXor = false)
{
    uint64_t state[128];

    std::memcpy(&state[0], refBlock, BLOCK_SIZE_BYTES);

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        state[i] ^= prevBlock[i];
    }

    for (int i = 0; i < BLOCK_SIZE; i += 16)
    {
        blamkaGeneric(
            state[i + 0],
            state[i + 1],
            state[i + 2],
            state[i + 3],
            state[i + 4],
            state[i + 5],
            state[i + 6],
            state[i + 7],
            state[i + 8],
            state[i + 9],
            state[i + 10],
            state[i + 11],
            state[i + 12],
            state[i + 13],
            state[i + 14],
            state[i + 15]
        );
    }

    for (int i = 0; i < BLOCK_SIZE / 8; i += 2)
    {
        blamkaGeneric(
            state[0 + i + 0],
            state[0 + i + 1],
            state[16 + i + 0],
            state[16 + i + 1],
            state[32 + i + 0],
            state[32 + i + 1],
            state[48 + i + 0],
            state[48 + i + 1],
            state[64 + i + 0],
            state[64 + i + 1],
            state[80 + i + 0],
            state[80 + i + 1],
            state[96 + i + 0],
            state[96 + i + 1],
            state[112 + i + 0],
            state[112 + i + 1]
        );
    }

    if (doXor)
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            nextBlock[i] ^= refBlock[i] ^ prevBlock[i] ^ state[i];
        }
    }
    else
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            nextBlock[i] = refBlock[i] ^ prevBlock[i] ^ state[i];
        }
    }
}

__device__ __forceinline__
void blake2bHash(
    uint8_t *out,
    uint8_t *input,
    uint32_t outputLength,
    uint32_t inputLength)
{
    size_t dataLength = sizeof(outputLength) + inputLength;

    /* Enough data for the output hash length and the input data */
    uint8_t *data = static_cast<uint8_t *>(malloc(dataLength));

    /* Prepend the length of the output hash length to the input data */
    std::memcpy(data, &outputLength, sizeof(outputLength));

    size_t offset = sizeof(outputLength);

    /* std::memcpy(data + sizeof(outputLength), input, inputLength); */
    for (int i = offset; i < dataLength; i++)
    {
        data[i] = input[i - offset];
    }

    /* Output length is less than 64 bytes, just hash it with blake and we're done */
    if (outputLength < HASH_SIZE)
    {
        blake2bGPU(out, data, dataLength, outputLength);

        /* Free memory */
        free(data);

        return;
    }

    /* Else make a HASH_SIZE buffer */
    uint8_t buffer[HASH_SIZE];

    /* Hash with blake into buffer */
    blake2bGPU(&buffer[0], data, dataLength, HASH_SIZE);

    /* Free memory */
    free(data);

    /* Copy the first 32 bytes to output */
    std::memcpy(out, &buffer[0], 32);

    out += 32;
    outputLength -= 32;

    while (outputLength > HASH_SIZE)
    {
        /* Repeatedly hash buffer data */
        blake2bGPU(&buffer[0], &buffer[0], HASH_SIZE, HASH_SIZE);

        /* And keep copying the first 32 bytes from buffer into the next 32
           bytes of output */
        std::memcpy(out, &buffer[0], 32);

        /* And repeat with the next 32 bytes */
        out += 32;
        outputLength -= 32;
    }

    if (outputLength % HASH_SIZE > 0)
    {
        uint32_t r = ((outputLength + 31) / 32) - 2;
        blake2bGPU(out, buffer, HASH_SIZE, outputLength - 32 * r);
    }
    else
    {
        blake2bGPU(out, buffer, HASH_SIZE, HASH_SIZE);
    }
}

__device__ __forceinline__
uint32_t indexAlpha(
    uint64_t random,
    uint32_t iteration,
    uint32_t slice,
    uint32_t index)
{
    uint32_t m = (3 * TRTL_SEGMENTS) + index - 1; 
    uint32_t s = ((slice + 1) % SYNC_POINTS) * TRTL_SEGMENTS;

    if (iteration == 0)
    {
        m = (slice * TRTL_SEGMENTS) + index - 1;
        s = 0;
    }

    uint64_t p = random & 0xFFFFFFFF;

    p = (p * p) >> 32;
    p = (p * m) >> 32;

    return static_cast<uint32_t>((s + m - (p + 1)) % static_cast<uint64_t>(TRTL_LANES));
}

__device__
void argon2idTRTLGPU(
    uint8_t *message,
    size_t messageLength, 
    uint8_t *salt,
    uint64_t *grid,
    uint8_t *result)
{
    /* STEP 1: INITIAL HASH */
    const uint32_t threads = 1;
    const uint32_t keyLen = RESULT_HASH_SIZE;
    const uint32_t memory = TRTL_MEMORY;
    const uint32_t time = TRTL_ITERATIONS;
    const uint32_t version = 19; /* Argon version */
    const uint32_t mode = 2; /* Argon2id */

    const uint32_t messageSize = static_cast<uint32_t>(messageLength);
    const uint32_t saltSize = TRTL_SALT_LENGTH;
    const uint32_t secretSize = 0;
    const uint32_t dataSize = 0;

    const size_t inputSize = sizeof(threads) + sizeof(keyLen) + sizeof(memory)
        + sizeof(time) + sizeof(version) + sizeof(mode) + sizeof(messageSize)
        + messageSize + sizeof(saltSize) + saltSize + sizeof(secretSize) + secretSize
        + sizeof(dataSize) + dataSize;

    uint8_t *initialInput = static_cast<uint8_t *>(malloc(inputSize));

    size_t index = 0;

    std::memcpy(&initialInput[index], &threads, sizeof(threads));
    index += sizeof(threads);

    std::memcpy(&initialInput[index], &keyLen, sizeof(keyLen));
    index += sizeof(keyLen);

    std::memcpy(&initialInput[index], &memory, sizeof(memory));
    index += sizeof(memory);

    std::memcpy(&initialInput[index], &time, sizeof(time));
    index += sizeof(time);

    std::memcpy(&initialInput[index], &version, sizeof(version));
    index += sizeof(version);

    std::memcpy(&initialInput[index], &mode, sizeof(mode));
    index += sizeof(mode);

    std::memcpy(&initialInput[index], &messageSize, sizeof(messageSize));
    index += sizeof(messageSize);

    std::memcpy(&initialInput[index], message, messageSize);
    index += messageSize;

    std::memcpy(&initialInput[index], &saltSize, sizeof(saltSize));
    index += sizeof(saltSize);

    std::memcpy(&initialInput[index], salt, saltSize);
    index += saltSize;

    std::memcpy(&initialInput[index], &secretSize, sizeof(secretSize));
    index += sizeof(secretSize);

    std::memcpy(&initialInput[index], &dataSize, sizeof(dataSize));
    index += sizeof(dataSize);

    uint8_t initialHash[INITIAL_HASH_SIZE] = {};

    blake2bGPU(initialHash, initialInput, inputSize, HASH_SIZE);

    free(initialInput);

    /* STEP 2: INIT BLOCKS */
    uint8_t block0[BLOCK_SIZE_BYTES];

    blake2bHash(block0, initialHash, BLOCK_SIZE_BYTES, INITIAL_HASH_SIZE);

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        /* grid[0][i] */
        grid[i] = *reinterpret_cast<uint64_t *>(&block0[i * 8]);
    }

    initialHash[64] = 1;

    blake2bHash(block0, initialHash, BLOCK_SIZE_BYTES, INITIAL_HASH_SIZE);

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        const size_t index = (1 * BLOCK_SIZE) + i;

        /* grid[1][i] */
        grid[index] = *reinterpret_cast<uint64_t *>(&block0[i * 8]);
    }

    /* STEP 3: PROCESS BLOCKS */

    /* Note: Since we only use one thread/lane for TRTL, I have replaced lane
       with zero where appropriate */
    for (int iteration = 0; iteration < TRTL_ITERATIONS; iteration++)
    {
        for (int slice = 0; slice < SYNC_POINTS; slice++)
        {
            uint64_t addresses[128] = {};
            uint64_t in[128] = {};
            uint64_t zero[128] = {};

            const bool modificationI = iteration == 0 && slice < SYNC_POINTS / 2;

            if (modificationI)
            {
                in[0] = iteration;
                in[2] = slice;
                in[3] = TRTL_SCRATCHPAD_SIZE;
                in[4] = TRTL_ITERATIONS;
                in[5] = 2; /* Argon2id */
            }

            uint32_t index = 0;

            if (iteration == 0 && slice == 0)
            {
                index = 2;
                in[6]++;
                processBlock(addresses, in, zero);
                processBlock(addresses, addresses, zero);
            }

            uint32_t offset = slice * TRTL_SEGMENTS + index;

            uint64_t random;

            while (index < TRTL_SEGMENTS)
            {
                uint32_t prev = offset - 1;

                if (index == 0 && slice == 0)
                {
                    prev += TRTL_LANES;
                }

                if (modificationI)
                {
                    const uint32_t addressIndex = index % BLOCK_SIZE;

                    if (addressIndex == 0)
                    {
                        in[6]++;
                        processBlock(addresses, in, zero);
                        processBlock(addresses, addresses, zero);
                    }

                    random = addresses[addressIndex];
                }
                else
                {
                    /* grid[prev][0] */
                    random = grid[prev * BLOCK_SIZE];
                }

                uint32_t newOffset = indexAlpha(random, iteration, slice, index);

                processBlock(
                    &grid[offset * BLOCK_SIZE],
                    &grid[prev * BLOCK_SIZE],
                    &grid[newOffset * BLOCK_SIZE],
                    true
                );

                index++;
                offset++;
            }
        }
    }
    
    /* STEP 4: EXTRACT KEY */
    for (uint32_t i = 0; i < BLOCK_SIZE; i++)
    {
        const size_t index = ((TRTL_SCRATCHPAD_SIZE - 1) * BLOCK_SIZE) + i;

        /* grid[TRTL_SCRATCHPAD_SIZE][i] */
        std::memcpy(&block0[i * 8], &grid[index], sizeof(uint64_t));
    }

    blake2bHash(result, block0, RESULT_HASH_SIZE, BLOCK_SIZE_BYTES);
}

void __global__
hashKernel(
    uint8_t *message,
    size_t messageLength, 
    uint8_t *salt,
    uint64_t *grid,
    uint8_t *result)
{
    argon2idTRTLGPU(message, messageLength, salt, grid, result);
}

/* input = 32 char byte array.
   output = 64 char hex string */
void byteArrayToHexString(const uint8_t *input, char *output)
{
    char hexval[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    for (int i = 0; i < RESULT_HASH_SIZE; i++)
    {
        output[i * 2] = hexval[((input[i] >> 4) & 0xF)];
        output[(i * 2) + 1] = hexval[(input[i]) & 0x0F];
    }
}

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::string errorStr = cudaGetErrorString(code);

        std::cout << "CUDA Error: " << errorStr << " at " << file << ", Line " << line << std::endl;

        if (abort)
        {
            throw std::runtime_error(errorStr);
        }
    }
}

std::vector<uint8_t> nvidiaHash(const std::vector<uint8_t> &input, const std::vector<uint8_t> &saltInput)
{
    size_t messageLength = input.size();

    uint8_t *message;
    uint8_t *salt;
    uint8_t *result;
    uint64_t *grid;

    /* Allocate message, salt, result, and grid on GPU memory */
    ERROR_CHECK(cudaMalloc((void **)&message, messageLength));
    ERROR_CHECK(cudaMalloc((void **)&salt, TRTL_SALT_LENGTH));
    ERROR_CHECK(cudaMalloc((void **)&result, RESULT_HASH_SIZE));
    ERROR_CHECK(cudaMalloc((void **)&grid, TRTL_SCRATCHPAD_SIZE * BLOCK_SIZE * sizeof(uint64_t)));

    /* Initialize message and salt on the GPU from the CPU */
    ERROR_CHECK(cudaMemcpy(message, &input[0], messageLength, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(salt, &saltInput[0], TRTL_SALT_LENGTH, cudaMemcpyHostToDevice));

    /* Zero out grid */
    ERROR_CHECK(cudaMemset(grid, 0, TRTL_SCRATCHPAD_SIZE * BLOCK_SIZE * sizeof(uint64_t)));

    /* Launch the kernel */
    hashKernel<<<1, 1>>>(message, messageLength, salt, grid, result);

    ERROR_CHECK(cudaPeekAtLastError());
    ERROR_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> hostResult(RESULT_HASH_SIZE);

    /* Copy the result from GPU memory to CPU memory */
    ERROR_CHECK(cudaMemcpy(&hostResult[0], result, RESULT_HASH_SIZE, cudaMemcpyDeviceToHost));

    cudaFree(message);
    cudaFree(salt);
    cudaFree(result);
    cudaFree(grid);

    return hostResult;
}
