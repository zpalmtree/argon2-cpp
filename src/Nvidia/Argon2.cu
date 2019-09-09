// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#include <cstring>
#include <stdint.h>
#include <iostream>
#include <vector>

#include "Argon2.h"
#include "Blake2.h"

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

/*
MIT License
Copyright (c) 2016 Ondrej Mosnáček
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
* Argon2d
* Simplified version of https://gitlab.com/omos/argon2-gpu
*/

__device__ uint64_t u64_build(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

__device__ uint32_t u64_lo(uint64_t x)
{
    return (uint32_t)x;
}

__device__ uint32_t u64_hi(uint64_t x)
{
    return (uint32_t)(x >> 32);
}

__device__ uint64_t u64_shuffle(uint64_t v, uint32_t thread)
{
    uint32_t lo = u64_lo(v);
    uint32_t hi = u64_hi(v);
    lo = __shfl_sync(0xFFFFFFFF, lo, thread);
    hi = __shfl_sync(0xFFFFFFFF, hi, thread);
    return u64_build(hi, lo);
}

struct block_l {
    uint32_t lo[ARGON_QWORDS_IN_BLOCK];
    uint32_t hi[ARGON_QWORDS_IN_BLOCK];
};

struct block_th {
    uint64_t a, b, c, d;
};

__device__ uint64_t cmpeq_mask(uint32_t test, uint32_t ref)
{
    uint32_t x = -(uint32_t)(test == ref);
    return u64_build(x, x);
}

__device__ uint64_t block_th_get(const struct block_th *b, uint32_t idx)
{
    uint64_t res = 0;
    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;
    return res;
}

__device__ void block_th_set(struct block_th *b, uint32_t idx, uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

__device__ void move_block(struct block_th *dst, const struct block_th *src)
{
    *dst = *src;
}

__device__ void xor_block(struct block_th *dst, const struct block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__ void load_block(struct block_th *dst, const struct block_g *src,
                           uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void load_block_xor(struct block_th *dst, const struct block_g *src,
                               uint32_t thread)
{
    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__ void store_block(struct block_g *dst, const struct block_th *src,
                            uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;
}

__device__ void block_l_store(struct block_l *dst, const struct block_th *src,
    uint32_t thread)
{
    dst->lo[0 * THREADS_PER_LANE + thread] = u64_lo(src->a);
    dst->hi[0 * THREADS_PER_LANE + thread] = u64_hi(src->a);

    dst->lo[1 * THREADS_PER_LANE + thread] = u64_lo(src->b);
    dst->hi[1 * THREADS_PER_LANE + thread] = u64_hi(src->b);

    dst->lo[2 * THREADS_PER_LANE + thread] = u64_lo(src->c);
    dst->hi[2 * THREADS_PER_LANE + thread] = u64_hi(src->c);

    dst->lo[3 * THREADS_PER_LANE + thread] = u64_lo(src->d);
    dst->hi[3 * THREADS_PER_LANE + thread] = u64_hi(src->d);
}

__device__ void block_l_load_xor(struct block_th *dst,
       const struct block_l *src, uint32_t thread)
{
    uint32_t lo, hi;

    lo = src->lo[0 * THREADS_PER_LANE + thread];
    hi = src->hi[0 * THREADS_PER_LANE + thread];
    dst->a ^= u64_build(hi, lo);

    lo = src->lo[1 * THREADS_PER_LANE + thread];
    hi = src->hi[1 * THREADS_PER_LANE + thread];
    dst->b ^= u64_build(hi, lo);

    lo = src->lo[2 * THREADS_PER_LANE + thread];
    hi = src->hi[2 * THREADS_PER_LANE + thread];
    dst->c ^= u64_build(hi, lo);

    lo = src->lo[3 * THREADS_PER_LANE + thread];
    hi = src->hi[3 * THREADS_PER_LANE + thread];
    dst->d ^= u64_build(hi, lo);
}

__device__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ uint64_t f(uint64_t x, uint64_t y)
{
    uint32_t xlo = u64_lo(x);
    uint32_t ylo = u64_lo(y);
    return x + y + 2 * u64_build(__umulhi(xlo, ylo), xlo * ylo);
}

__device__ void g(struct block_th *block)
{
    uint64_t a, b, c, d;
    a = block->a;
    b = block->b;
    c = block->c;
    d = block->d;

    a = f(a, b);
    d = rotr64(d ^ a, 32);
    c = f(c, d);
    b = rotr64(b ^ c, 24);
    a = f(a, b);
    d = rotr64(d ^ a, 16);
    c = f(c, d);
    b = rotr64(b ^ c, 63);

    block->a = a;
    block->b = b;
    block->c = c;
    block->d = d;
}

template<class shuffle>
__device__ void apply_shuffle(struct block_th *block, uint32_t thread)
{
    for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
        uint32_t src_thr = shuffle::apply(thread, i);

        uint64_t v = block_th_get(block, i);
        v = u64_shuffle(v, src_thr);
        block_th_set(block, i, v);
    }
}

__device__ void transpose(struct block_th *block, uint32_t thread)
{
    uint32_t thread_group = (thread & 0x0C) >> 2;
    for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
        uint32_t thr = (i << 2) ^ thread;
        uint32_t idx = thread_group ^ i;

        uint64_t v = block_th_get(block, idx);
        v = u64_shuffle(v, thr);
        block_th_set(block, idx, v);
    }
}

struct identity_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return thread;
    }
};

struct shift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct unshift1_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        return (thread & 0x1c) | ((thread + idx) & 0x3);
    }
};

struct shift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

struct unshift2_shuffle {
    __device__ static uint32_t apply(uint32_t thread, uint32_t idx)
    {
        idx = (QWORDS_PER_THREAD - idx) % QWORDS_PER_THREAD;

        uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
        lo = (lo + idx) & 0x3;
        return ((lo & 0x2) << 3) | (thread & 0xe) | (lo & 0x1);
    }
};

__device__ void shuffle_block(struct block_th *block, uint32_t thread)
{
    transpose(block, thread);

    g(block);

    apply_shuffle<shift1_shuffle>(block, thread);

    g(block);

    apply_shuffle<unshift1_shuffle>(block, thread);
    transpose(block, thread);

    g(block);

    apply_shuffle<shift2_shuffle>(block, thread);

    g(block);

    apply_shuffle<unshift2_shuffle>(block, thread);
}

__device__ void compute_ref_pos(uint32_t offset, uint32_t *ref_index)
{
    uint32_t ref_area_size = offset - 1;
    *ref_index = __umulhi(*ref_index, *ref_index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);
}

__device__ void argon2_core(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_l *tmp,
        uint32_t thread, uint32_t ref_index)
{
    struct block_g *mem_ref = memory + ref_index;

    load_block_xor(prev, mem_ref, thread);
    block_l_store(tmp, prev, thread);

    shuffle_block(prev, thread);

    block_l_load_xor(prev, tmp, thread);

    store_block(mem_curr, prev, thread);
}

__device__ void argon2_step(
        struct block_g *memory, struct block_g *mem_curr,
        struct block_th *prev, struct block_l *tmp,
        uint32_t thread, uint32_t offset)
{
    uint64_t v = u64_shuffle(prev->a, 0);
    uint32_t ref_index = u64_lo(v);

    compute_ref_pos(offset, &ref_index);

    argon2_core(memory, mem_curr, prev, tmp, thread, ref_index);
}

__global__
void argon2Kernel(
    struct block_g *memory,
    uint32_t memory_cost)
{
    extern __shared__ struct block_l shared[];

    uint32_t job_id = blockIdx.y;
    uint32_t thread = threadIdx.x;

    /* select job's memory region: */
    memory += (size_t)job_id * memory_cost;

    struct block_th prev;
    struct block_l *tmp = &shared[0];

    struct block_g *mem_lane = memory;
    struct block_g *mem_prev = mem_lane + 1;
    struct block_g *mem_curr = mem_lane + 2;

    load_block(&prev, mem_prev, thread);

    uint32_t skip = 2;

    for (uint32_t offset = 2; offset < memory_cost; ++offset) {
        argon2_step(memory, mem_curr, &prev, tmp, thread, offset);
        mem_curr++;
    }
}

kernelLaunchParams getLaunchParams(
    const uint32_t gpuIndex)
{
    kernelLaunchParams params;

    cudaDeviceProp properties;

    /* Figure out how much memory we have available */
    cudaGetDeviceProperties(&properties, gpuIndex);

    const size_t ONE_MB = 1024 * 1024;
    const size_t ONE_GB = ONE_MB * 1024;

    size_t memoryAvailable = (properties.totalGlobalMem / ONE_GB - 1) * (ONE_GB / ONE_MB);

    /* The amount of nonces we're going to try per kernel launch */
    uint32_t noncesPerRun = (memoryAvailable * ONE_MB) / (sizeof(block_g) * TRTL_SCRATCHPAD_SIZE);
    noncesPerRun = (noncesPerRun / BLAKE_THREADS_PER_BLOCK) * BLAKE_THREADS_PER_BLOCK;

    /* The amount of memory we'll need to allocate on the GPU */
    params.memSize = sizeof(block_g) * TRTL_MEMORY * noncesPerRun;

    /* Init memory kernel params */
    params.initMemoryBlocks = noncesPerRun / BLAKE_THREADS_PER_BLOCK;
    params.initMemoryThreads = BLAKE_THREADS_PER_BLOCK;

    /* Argon2 kernel params */
    params.argon2Blocks = noncesPerRun;
    params.argon2Threads = THREADS_PER_LANE;
    params.argon2Cache = params.cache * ARGON_BLOCK_SIZE;

    params.getNonceBlocks = noncesPerRun / BLAKE_THREADS_PER_BLOCK;
    params.getNonceThreads = BLAKE_THREADS_PER_BLOCK;

    params.noncesPerRun = noncesPerRun;

    return params;
}

/**
 * Stuff we only need to do once (unless the algorithm changes).
 */
NvidiaState initializeState(const uint32_t gpuIndex)
{
    /* Set current device */
    ERROR_CHECK(cudaSetDevice(gpuIndex));

    NvidiaState state;

    state.launchParams = getLaunchParams(gpuIndex);

    /* Allocate memory. These things will the be the same size for every job,
       unless the algorithm changes. */
    ERROR_CHECK(cudaMalloc((void **)&state.memory, state.launchParams.memSize));
    ERROR_CHECK(cudaMalloc((void **)&state.nonce, sizeof(uint32_t)));
    ERROR_CHECK(cudaMalloc((void **)&state.hash, ARGON_HASH_LENGTH));
    ERROR_CHECK(cudaMalloc((void **)&state.hashFound, sizeof(bool)));
    ERROR_CHECK(cudaMalloc((void **)&state.blakeInput, BLAKE_BLOCK_SIZE * 2));

    ERROR_CHECK(cudaMemset(state.hashFound, false, sizeof(bool)));
    ERROR_CHECK(cudaMemset(state.nonce, 0, sizeof(uint32_t)));

    return state;
}

void freeState(NvidiaState &state)
{
    ERROR_CHECK(cudaFree(state.memory));
    ERROR_CHECK(cudaFree(state.nonce));
    ERROR_CHECK(cudaFree(state.hash));
    ERROR_CHECK(cudaFree(state.hashFound));
    ERROR_CHECK(cudaFree(state.blakeInput));
}

void initJob(
    NvidiaState &state,
    const std::vector<uint8_t> &input,
    const std::vector<uint8_t> &saltInput,
    const uint32_t localNonce,
    const uint64_t target)
{
    state.localNonce = localNonce;
    state.target = target;

    setupBlakeInput(input, saltInput, state);
}

HashResult nvidiaHash(NvidiaState &state)
{
    /* Launch the first kernel to perform initial blake initialization */
    initMemoryKernel<<<
        dim3(state.launchParams.initMemoryBlocks),
        dim3(state.launchParams.initMemoryThreads)
    >>>(
        state.memory,
        state.blakeInput,
        TRTL_MEMORY,
        state.localNonce,
        state.blakeInputSize
    );

    /* Launch the second kernel to perform the main argon work */
    argon2Kernel<<<
        dim3(1, state.launchParams.argon2Blocks),
        dim3(state.launchParams.argon2Threads, 1),
        ARGON_BLOCK_SIZE
    >>>(
        state.memory,
        TRTL_MEMORY
    );

    /* Launch the final kernel to perform final blake round and extract
       nonce that beats the target, if any */
    getNonceKernel<<<
        dim3(state.launchParams.getNonceBlocks),
        dim3(state.launchParams.getNonceThreads)
    >>>(
        state.memory,
        state.localNonce,
        state.target,
        state.nonce,
        state.hash,
        state.hashFound
    );

    /* Wait for kernel */
    ERROR_CHECK(cudaPeekAtLastError());
    ERROR_CHECK(cudaDeviceSynchronize());

    HashResult result;

    /* See if we found a valid nonce */
    ERROR_CHECK(cudaMemcpy(&result.success, state.hashFound, sizeof(result.success), cudaMemcpyDeviceToHost));
    
    if (result.success)
    {   
        /* Copy valid nonce + hash back to CPU */
        ERROR_CHECK(cudaMemcpy(&result.nonce, state.nonce, sizeof(result.nonce), cudaMemcpyDeviceToHost));
        ERROR_CHECK(cudaMemcpy(&result.hash, state.hash, ARGON_HASH_LENGTH, cudaMemcpyDeviceToHost));

        /* Clear the hash found flag so don't think we have found a share when we
           have not, along with the nonce */
        ERROR_CHECK(cudaMemset(state.hashFound, false, sizeof(bool)));
        ERROR_CHECK(cudaMemset(state.nonce, 0, sizeof(uint32_t)));
    }
    
    return result;
}
