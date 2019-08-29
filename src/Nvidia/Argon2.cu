// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include <cstring>
#include <stdint.h>
#include <iostream>

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
const size_t SYNC_POINTS = 4;
const size_t HASH_SIZE = 64;
const size_t INITIAL_HASH_SIZE = 72;

const uint32_t TRTL_MEMORY = 512;
const uint32_t TRTL_SCRATCHPAD_SIZE = TRTL_MEMORY;
const uint32_t TRTL_LANES = TRTL_MEMORY;
const uint32_t TRTL_SALT_LENGTH = 16;
const uint32_t TRTL_ITERATIONS = 3;

__device__
void argon2idTRTLGPU(
    uint8_t *message,
    size_t messageLength, 
    uint8_t *salt,
    uint8_t *result)
{
    uint64_t grid[TRTL_SCRATCHPAD_SIZE][BLOCK_SIZE] = {};

    const uint32_t threads = 1;
    const uint32_t keyLen = 32;
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

    uint8_t *initialInput = (uint8_t *)malloc(inputSize);

    std::memcpy(&initialInput[0], &threads, sizeof(threads));
    std::memcpy(&initialInput[4], &keyLen, sizeof(keyLen));
    std::memcpy(&initialInput[8], &memory, sizeof(memory));
    std::memcpy(&initialInput[12], &time, sizeof(time));
    std::memcpy(&initialInput[16], &version, sizeof(version));
    std::memcpy(&initialInput[20], &mode, sizeof(mode));
    std::memcpy(&initialInput[24], &message, messageSize * sizeof(uint8_t));
    std::memcpy(&initialInput[24 + messageSize], &salt, saltSize * sizeof(uint8_t));

    uint8_t initialHash[INITIAL_HASH_SIZE];

    blake2bGPU(initialHash, initialInput, inputSize, HASH_SIZE);

    free(initialInput);

    for (int i = 0; i < 32; i++)
    {
        result[i] = initialHash[i];
    }
}

void __global__
hashKernel(
    uint8_t *message,
    size_t messageLength, 
    uint8_t *salt,
    uint8_t *result)
{
    argon2idTRTLGPU(message, messageLength, salt, result);
}

/* input = 32 char byte array.
   output = 64 char hex string */
void byteArrayToHexString(const uint8_t *input, char *output)
{
    char hexval[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    for (int i = 0; i < 32; i++)
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

        std::cout << "CUDA Error " << errorStr << " at " << file << ", Line " << line << std::endl;

        if (abort)
        {
            throw std::runtime_error(errorStr);
        }
    }
}

void hash()
{
    const uint8_t chukwaInput[] = {
        1, 0, 251, 142, 138, 200, 5, 137, 147, 35, 55, 27, 183, 144, 219, 25,
        33, 138, 253, 141, 184, 227, 117, 93, 139, 144, 243, 155, 61, 85, 6,
        169, 171, 206, 79, 169, 18, 36, 69, 0, 0, 0, 0, 238, 129, 70, 212, 159,
        169, 62, 231, 36, 222, 181, 125, 18, 203, 198, 198, 243, 185, 36, 217,
        70, 18, 124, 122, 151, 65, 143, 147, 72, 130, 143, 15, 2
    };

    size_t messageLength = sizeof(chukwaInput) / sizeof(*chukwaInput);

    uint8_t *message;
    uint8_t *salt;
    uint8_t *result;

    /* Allocate message, salt, and result on GPU memory */
    ERROR_CHECK(cudaMalloc((void **)&message, messageLength * sizeof(uint8_t)));
    ERROR_CHECK(cudaMalloc((void **)&salt, 16 * sizeof(uint8_t)));
    ERROR_CHECK(cudaMalloc((void **)&result, 32 * sizeof(uint8_t)));

    /* Initialize message and salt on the GPU from the CPU */
    ERROR_CHECK(cudaMemcpy(message, &chukwaInput, messageLength, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(salt, &chukwaInput, 16, cudaMemcpyHostToDevice));

    std::cout << "Launching kernel" << std::endl;

    /* Launch the kernel */
    hashKernel<<<1, 1>>>(message, messageLength, salt, result);

    ERROR_CHECK(cudaPeekAtLastError());
    ERROR_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel finished running" << std::endl;

    uint8_t hostResult[32];

    /* Copy the result from GPU memory to CPU memory */
    ERROR_CHECK(cudaMemcpy(&hostResult, result, 32, cudaMemcpyDeviceToHost));

    char output[65];

    byteArrayToHexString(hostResult, output);

    output[64] = '\0';

    std::cout << output << std::endl;

    cudaFree(message);
    cudaFree(salt);
    cudaFree(result);
}
