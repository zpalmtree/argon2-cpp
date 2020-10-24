// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <vector>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

const size_t ARGON_BLOCK_SIZE = 1024;
const size_t ARGON_QWORDS_IN_BLOCK = ARGON_BLOCK_SIZE / 8;
const size_t ARGON_HASH_LENGTH = 32;
const size_t ARGON_SYNC_POINTS = 4;

const size_t BLAKE_BLOCK_SIZE = 128;
const size_t BLAKE_DWORDS_IN_BLOCK = BLAKE_BLOCK_SIZE / 4;
const size_t BLAKE_QWORDS_IN_BLOCK = BLAKE_BLOCK_SIZE / 8;
const size_t BLAKE_HASH_LENGTH = 64;
const size_t BLAKE_THREADS_PER_BLOCK = 128;
const size_t BLAKE_INITIAL_HASH_LENGTH = 76;

const uint32_t THREADS_PER_LANE = 32;
const size_t QWORDS_PER_THREAD = ARGON_QWORDS_IN_BLOCK / THREADS_PER_LANE;

struct block_g
{
    uint64_t data[ARGON_QWORDS_IN_BLOCK];
};

struct HashResult
{
    /* The nonce of the valid hash */
    uint32_t nonce;

    /* Did we find a valid hash? */
    bool success = false;

    /* The valid hash, if we found one */
    uint8_t hash[32];
};

struct kernelLaunchParams
{
    size_t memSize;

    dim3 initMemoryBlocks;
    dim3 initMemoryThreads;

    dim3 argon2Blocks;
    dim3 argon2Threads;
    size_t argon2Cache;

    dim3 getNonceBlocks;
    dim3 getNonceThreads;

    size_t noncesPerRun;

    size_t jobsPerBlock;

    size_t scratchpadSize;
    size_t iterations;
};

struct NvidiaState
{
    /* Allocated once per algorithm */

    /* Scratchpad, stored on GPU */
    block_g *memory = NULL;

    /* Nonce, stored on GPU */
    uint32_t *nonce = NULL;

    /* Final hash, stored on GPU */
    uint8_t *hash = NULL;

    /* Whether we found a hash, stored on GPU */
    bool *hashFound = NULL;

    /* Params to launch each kernel with */
    kernelLaunchParams launchParams;

    /* Allocated once per job */

    /* Size of the message input once the argon params are appended. */
    uint32_t blakeInputSize;

    /* Message + salt + argon params */
    uint64_t *blakeInput = NULL;

    /* Nonce to begin hashing at */
    uint32_t localNonce;

    /* Target hash needs to meet */
    uint64_t target;

    /* Whether we should use nicehash style nonces */
    bool isNiceHash;

    cudaStream_t stream = NULL;
};

inline bool throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
    if (code == cudaErrorUnknown)
    {
        std::cout << "Recieved cudaErrorUnknown (999) from Nvidia device. Your PC may need restarting." << std::endl;
        return false;
    }

    if (code == cudaErrorDevicesUnavailable)
    {
        std::cout << "Recieved cudaErrorDevicesUnavailable from Nvidia device. Another application may be using the GPU exclusively." << std::endl;
        return false;
    }

    if (code == cudaErrorInitializationError)
    {
        std::cout << "Received cudaErrorInitializationError from Nvidia device. You may need to update your GPU drivers." << std::endl;
        return false;
    }

    if (code == cudaErrorInsufficientDriver)
    {
        return false;
    }

    if (code != cudaSuccess)
    {
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }

    return true;
}

NvidiaState initializeState(
    const uint32_t gpuIndex,
    const size_t scratchpadSize,
    const size_t iterations,
    const float intensity,
    uint32_t attempt = 0);

void freeState(NvidiaState &state);

void initJob(
    NvidiaState &state,
    const std::vector<uint8_t> &input,
    const std::vector<uint8_t> &saltInput,
    const uint64_t target);

HashResult nvidiaHash(NvidiaState &state);
