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

// Portions Copyright (c) 2018 tomkha

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


__device__
uint64_t u64_build(
    const uint32_t hi,
    const uint32_t lo)
{
    return static_cast<uint64_t>(hi) << 32 | static_cast<uint64_t>(lo);
}

__device__
uint32_t u64_lo(uint64_t x)
{
    return static_cast<uint32_t>(x);
}

__device__
uint32_t u64_hi(uint64_t x)
{
    return static_cast<uint32_t>(x >> 32);
}

struct u64_shuffle_buf
{
    uint32_t lo[THREADS_PER_LANE];
    uint32_t hi[THREADS_PER_LANE];
};

__device__
uint64_t u64_shuffle(
    const uint64_t v,
    const uint32_t thread_src,
    const uint32_t thread,
    u64_shuffle_buf *buf)
{
    uint32_t lo = u64_lo(v);
    uint32_t hi = u64_hi(v);

    buf->lo[thread] = lo;
    buf->hi[thread] = hi;

    __syncthreads();

    lo = buf->lo[thread_src];
    hi = buf->hi[thread_src];

    return u64_build(hi, lo);
}

struct block_l
{
    uint32_t lo[ARGON_QWORDS_IN_BLOCK];
    uint32_t hi[ARGON_QWORDS_IN_BLOCK];
};

struct block_th
{
    uint64_t a, b, c, d;
};

__device__
uint64_t cmpeq_mask(
    const uint32_t test,
    const uint32_t ref)
{
    uint32_t x = -static_cast<uint32_t>(test == ref);

    return u64_build(x, x);
}

__device__
uint64_t block_th_get(
    const block_th *b,
    const uint32_t idx)
{
    uint64_t res = 0;

    res ^= cmpeq_mask(idx, 0) & b->a;
    res ^= cmpeq_mask(idx, 1) & b->b;
    res ^= cmpeq_mask(idx, 2) & b->c;
    res ^= cmpeq_mask(idx, 3) & b->d;

    return res;
}

__device__
void block_th_set(
    block_th *b,
    const uint32_t idx,
    const uint64_t v)
{
    b->a ^= cmpeq_mask(idx, 0) & (v ^ b->a);
    b->b ^= cmpeq_mask(idx, 1) & (v ^ b->b);
    b->c ^= cmpeq_mask(idx, 2) & (v ^ b->c);
    b->d ^= cmpeq_mask(idx, 3) & (v ^ b->d);
}

__device__
void move_block(
    block_th *dst,
    const block_th *src)
{
    *dst = *src;
}

__device__
void xor_block(
    block_th *dst,
    const block_th *src)
{
    dst->a ^= src->a;
    dst->b ^= src->b;
    dst->c ^= src->c;
    dst->d ^= src->d;
}

__device__
void load_block(
    block_th *dst,
    const block_g *src,
    const uint32_t thread)
{
    dst->a = src->data[0 * THREADS_PER_LANE + thread];
    dst->b = src->data[1 * THREADS_PER_LANE + thread];
    dst->c = src->data[2 * THREADS_PER_LANE + thread];
    dst->d = src->data[3 * THREADS_PER_LANE + thread];
}

__device__
void load_block_xor(
    block_th *dst,
    const block_g *src,
    const uint32_t thread)
{
    dst->a ^= src->data[0 * THREADS_PER_LANE + thread];
    dst->b ^= src->data[1 * THREADS_PER_LANE + thread];
    dst->c ^= src->data[2 * THREADS_PER_LANE + thread];
    dst->d ^= src->data[3 * THREADS_PER_LANE + thread];
}

__device__
void store_block(
    block_g *dst,
    const block_th *src,
    const uint32_t thread)
{
    dst->data[0 * THREADS_PER_LANE + thread] = src->a;
    dst->data[1 * THREADS_PER_LANE + thread] = src->b;
    dst->data[2 * THREADS_PER_LANE + thread] = src->c;
    dst->data[3 * THREADS_PER_LANE + thread] = src->d;
}

__device__
void block_l_store(
    block_l *dst,
    struct block_th *src,
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

__device__
void block_l_load_xor(
    block_th *dst,
    const block_l *src,
    uint32_t thread)
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

__device__
void g(block_th *block)
{
    asm("{"
        ".reg .u64 s, x;"
        ".reg .u32 l1, l2, h1, h2;"
        // a = f(a, b);
        "add.u64 s, %0, %1;"            // s = a + b
        "cvt.u32.u64 l1, %0;"           // xlo = u64_lo(a)
        "cvt.u32.u64 l2, %1;"           // ylo = u64_lo(b)
        "mul.hi.u32 h1, l1, l2;"        // umulhi(xlo, ylo)
        "mul.lo.u32 l1, l1, l2;"        // xlo * ylo
        "mov.b64 x, {l1, h1};"          // x = u64_build(umulhi(xlo, ylo), xlo * ylo)
        "shl.b64 x, x, 1;"              // x = 2 * x
        "add.u64 %0, s, x;"             // a = s + x
        // d = rotr64(d ^ a, 32);
        "xor.b64 x, %3, %0;"
        "mov.b64 {h2, l2}, x;"
        "mov.b64 %3, {l2, h2};"         // swap hi and lo = rotr64(x, 32)
        // c = f(c, d);
        "add.u64 s, %2, %3;"
        "cvt.u32.u64 l1, %2;"
        "mul.hi.u32 h1, l1, l2;"
        "mul.lo.u32 l1, l1, l2;"
        "mov.b64 x, {l1, h1};"
        "shl.b64 x, x, 1;"
        "add.u64 %2, s, x;"
        // b = rotr64(b ^ c, 24);
        "xor.b64 x, %1, %2;"
        "mov.b64 {l1, h1}, x;"
        "prmt.b32 l2, l1, h1, 0x6543;"  // permute bytes 76543210 => 21076543
        "prmt.b32 h2, l1, h1, 0x2107;"  // rotr64(x, 24)
        "mov.b64 %1, {l2, h2};"
        // a = f(a, b);
        "add.u64 s, %0, %1;"
        "cvt.u32.u64 l1, %0;"
        "mul.hi.u32 h1, l1, l2;"
        "mul.lo.u32 l1, l1, l2;"
        "mov.b64 x, {l1, h1};"
        "shl.b64 x, x, 1;"
        "add.u64 %0, s, x;"
        // d = rotr64(d ^ a, 16);
        "xor.b64 x, %3, %0;"
        "mov.b64 {l1, h1}, x;"
        "prmt.b32 l2, l1, h1, 0x5432;"  // permute bytes 76543210 => 10765432
        "prmt.b32 h2, l1, h1, 0x1076;"  // rotr64(x, 16)
        "mov.b64 %3, {l2, h2};"
        // c = f(c, d);
        "add.u64 s, %2, %3;"
        "cvt.u32.u64 l1, %2;"
        "mul.hi.u32 h1, l1, l2;"
        "mul.lo.u32 l1, l1, l2;"
        "mov.b64 x, {l1, h1};"
        "shl.b64 x, x, 1;"
        "add.u64 %2, s, x;"
        // b = rotr64(b ^ c, 63);
        "xor.b64 x, %1, %2;"
        "shl.b64 s, x, 1;"              // x << 1
        "shr.b64 x, x, 63;"             // x >> 63
        "add.u64 %1, s, x;"             // emits less instructions than "or"
        "}"
        : "+l"(block->a), "+l"(block->b), "+l"(block->c), "+l"(block->d)
    );
}

__device__ void transpose(
    block_th *block,
    const uint32_t thread)
{
    // thread groups, previously: thread_group = (thread & 0x0C) >> 2
    const uint32_t g1 = (thread & 0x4);
    const uint32_t g2 = (thread & 0x8);

    uint64_t x1 = (g2 ? (g1 ? block->c : block->d) : (g1 ? block->a : block->b));
    uint64_t x2 = (g2 ? (g1 ? block->b : block->a) : (g1 ? block->d : block->c));
    uint64_t x3 = (g2 ? (g1 ? block->a : block->b) : (g1 ? block->c : block->d));

    x1 = __shfl_xor_sync(0xFFFFFFFF, x1, 0x4);
    x2 = __shfl_xor_sync(0xFFFFFFFF, x2, 0x8);
    x3 = __shfl_xor_sync(0xFFFFFFFF, x3, 0xC);

    block->a = (g2 ? (g1 ? x3 : x2) : (g1 ? x1 : block->a));
    block->b = (g2 ? (g1 ? x2 : x3) : (g1 ? block->b : x1));
    block->c = (g2 ? (g1 ? x1 : block->c) : (g1 ? x3 : x2));
    block->d = (g2 ? (g1 ? block->d : x1) : (g1 ? x2 : x3));
}

__device__
void shift1_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t src_thr_b = (thread & 0x1c) | ((thread + 1) & 0x3);
    const uint32_t src_thr_d = (thread & 0x1c) | ((thread + 3) & 0x3);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__
void unshift1_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t src_thr_b = (thread & 0x1c) | ((thread + 3) & 0x3);
    const uint32_t src_thr_d = (thread & 0x1c) | ((thread + 1) & 0x3);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__
void shift2_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    const uint32_t src_thr_b = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);
    const uint32_t src_thr_d = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__
void unshift2_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    const uint32_t src_thr_b = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);
    const uint32_t src_thr_d = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);

    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
}

__device__ void shuffle_block(
    block_th *block,
    const uint32_t thread)
{
    transpose(block, thread);

    g(block);

    shift1_shuffle(block, thread);

    g(block);

    unshift1_shuffle(block, thread);
    transpose(block, thread);

    g(block);

    shift2_shuffle(block, thread);

    g(block);

    unshift2_shuffle(block, thread);
}

__device__
void argon2_core(
    block_g *memory,
    block_g *mem_curr,
    block_th *prev,
    block_th *tmp,
    const uint32_t thread,
    const uint32_t pass,
    const uint32_t ref_index,
    const uint32_t ref_lane)
{
    block_g *mem_ref = memory + ref_index + ref_lane;

    if (pass != 0)
    {
        load_block(tmp, mem_curr, thread);
        load_block_xor(prev, mem_ref, thread);
        xor_block(tmp, prev);
    }
    else
    {
        load_block_xor(prev, mem_ref, thread);
        move_block(tmp, prev);
    }

    shuffle_block(prev, thread);

    xor_block(prev, tmp);

    store_block(mem_curr, prev, thread);
}

__device__
void next_addresses(
    block_th *addr,
    block_th *tmp,
    const uint32_t thread_input,
    const uint32_t thread)
{
    addr->a = u64_build(0, thread_input);
    addr->b = 0;
    addr->c = 0;
    addr->d = 0;

    shuffle_block(addr, thread);

    addr->a ^= u64_build(0, thread_input);
    move_block(tmp, addr);

    shuffle_block(addr, thread);

    xor_block(addr, tmp);
}

__device__
void compute_ref_pos(
    const uint32_t segment_blocks,
    const uint32_t pass,
    const uint32_t slice,
    const uint32_t offset,
    uint32_t *ref_lane,
    uint32_t *ref_index)
{
    const uint32_t lane_blocks = ARGON_SYNC_POINTS * segment_blocks;

    *ref_lane = 0;

    uint32_t base;

    if (pass != 0)
    {
        base = lane_blocks - segment_blocks;
    }
    else
    {
        base = slice * segment_blocks;
    }

    uint32_t ref_area_size = base + offset - 1;

    *ref_index = __umulhi(*ref_index, *ref_index);
    *ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);

    if (pass != 0 && slice != ARGON_SYNC_POINTS - 1)
    {
        *ref_index += (slice + 1) * segment_blocks;

        if (*ref_index >= lane_blocks)
        {
            *ref_index -= lane_blocks;
        }
    }
}

__device__
void argon2_step(
    block_g *memory,
    block_g *mem_curr,
    block_th *prev,
    block_th *tmp,
    block_th *addr,
    u64_shuffle_buf *shuffle_buf,
    const uint32_t segment_blocks,
    const uint32_t thread,
    uint32_t *thread_input,
    const uint32_t pass,
    const uint32_t slice,
    const uint32_t offset)
{
    uint32_t ref_index, ref_lane;

    if (pass == 0 && slice < ARGON_SYNC_POINTS / 2)
    {
        uint32_t addr_index = offset % ARGON_QWORDS_IN_BLOCK;

        if (addr_index == 0)
        {
            if (thread == 6)
            {
                ++*thread_input;
            }

            next_addresses(addr, tmp, *thread_input, thread);
        }

        uint32_t thr = addr_index % THREADS_PER_LANE;
        uint32_t idx = addr_index / THREADS_PER_LANE;

        uint64_t v = block_th_get(addr, idx);

        v = u64_shuffle(v, thr, thread, shuffle_buf);

        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    }
    else
    {
        uint64_t v = u64_shuffle(prev->a, 0, thread, shuffle_buf);
        ref_index = u64_lo(v);
        ref_lane  = u64_hi(v);
    }

    compute_ref_pos(
        segment_blocks,
        pass,
        slice,
        offset,
        &ref_lane,
        &ref_index
    );

    argon2_core(
        memory,
        mem_curr,
        prev,
        tmp,
        thread,
        pass,
        ref_index,
        ref_lane
    );
}

__global__
void argon2Kernel(
    block_g *memory,
    const uint32_t passes,
    const uint32_t segment_blocks)
{
    extern __shared__ u64_shuffle_buf shuffle_bufs[];
    u64_shuffle_buf *shuffle_buf = &shuffle_bufs[threadIdx.z + threadIdx.y];

    uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t thread = threadIdx.x;

    uint32_t lane_blocks = ARGON_SYNC_POINTS * segment_blocks;

    /* select job's memory region: */
    memory += static_cast<size_t>(job_id) * lane_blocks;

    struct block_th prev, addr, tmp;

    uint32_t thread_input = 0;

    switch (thread)
    {
        case 3:
        {
            thread_input = lane_blocks;
            break;
        }
        case 4:
        {
            thread_input = passes;
            break;
        }
        case 5:
        {
            thread_input = 2; // Argon2id
            break;
        }
    }

    if (segment_blocks > 2)
    {
        if (thread == 6)
        {
            ++thread_input;
        }

        next_addresses(&addr, &tmp, thread_input, thread);
    }

    block_g *mem_lane = memory;
    block_g *mem_prev = mem_lane + 1;
    block_g *mem_curr = mem_lane + 2;

    load_block(&prev, mem_prev, thread);

    uint32_t skip = 2;

    for (uint32_t pass = 0; pass < passes; pass++)
    {
        for (uint32_t slice = 0; slice < ARGON_SYNC_POINTS; slice++)
        {
            for (uint32_t offset = 0; offset < segment_blocks; offset++)
            {
                if (skip > 0)
                {
                    --skip;
                    continue;
                }

                argon2_step(
                    memory,
                    mem_curr,
                    &prev,
                    &tmp,
                    &addr,
                    shuffle_buf,
                    segment_blocks,
                    thread,
                    &thread_input,
                    pass,
                    slice,
                    offset
                );

                mem_curr++;
            }

            __syncthreads();

            if (thread == 2)
            {
                ++thread_input;
            }
            else if (thread == 6)
            {
                thread_input = 0;
            }
        }

        mem_curr = mem_lane;
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

    params.jobsPerBlock = 16;

    /* Argon2 kernel params */
    params.argon2Blocks = noncesPerRun / params.jobsPerBlock;
    params.argon2Threads = THREADS_PER_LANE;
    params.argon2Cache = params.jobsPerBlock * sizeof(u64_shuffle_buf);

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
        dim3(state.launchParams.initMemoryThreads, 2)
    >>>(
        state.memory,
        state.blakeInput,
        state.blakeInputSize,
        state.localNonce
    );

    /* Launch the second kernel to perform the main argon work */
    argon2Kernel<<<
        dim3(1, 1, state.launchParams.argon2Blocks),
        dim3(state.launchParams.argon2Threads, 1, state.launchParams.jobsPerBlock),
        state.launchParams.argon2Cache
    >>>(
        state.memory,
        TRTL_ITERATIONS,
        TRTL_MEMORY / ARGON_SYNC_POINTS
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
