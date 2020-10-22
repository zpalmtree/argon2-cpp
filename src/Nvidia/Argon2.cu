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
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

#include "Argon2.h"
#include "Blake2.h"

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

    #if CUDART_VERSION < 9000
    x1 = __shfl_xor(x1, 0x4);
    x2 = __shfl_xor(x2, 0x8);
    x3 = __shfl_xor(x3, 0xC);
    #else
    x1 = __shfl_xor_sync(0xFFFFFFFF, x1, 0x4);
    x2 = __shfl_xor_sync(0xFFFFFFFF, x2, 0x8);
    x3 = __shfl_xor_sync(0xFFFFFFFF, x3, 0xC);
    #endif

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

    #if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x2);
    block->d = __shfl(block->d, src_thr_d);
    #else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
    #endif
}

__device__
void unshift1_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t src_thr_b = (thread & 0x1c) | ((thread + 3) & 0x3);
    const uint32_t src_thr_d = (thread & 0x1c) | ((thread + 1) & 0x3);

    #if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x2);
    block->d = __shfl(block->d, src_thr_d);
    #else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x2);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
    #endif
}

__device__
void shift2_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    const uint32_t src_thr_b = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);
    const uint32_t src_thr_d = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);

    #if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x10);
    block->d = __shfl(block->d, src_thr_d);
    #else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
    #endif
}

__device__
void unshift2_shuffle(
    block_th *block,
    const uint32_t thread)
{
    const uint32_t lo = (thread & 0x1) | ((thread & 0x10) >> 3);
    const uint32_t src_thr_b = (((lo + 3) & 0x2) << 3) | (thread & 0xe) | ((lo + 3) & 0x1);
    const uint32_t src_thr_d = (((lo + 1) & 0x2) << 3) | (thread & 0xe) | ((lo + 1) & 0x1);

    #if CUDART_VERSION < 9000
    block->b = __shfl(block->b, src_thr_b);
    block->c = __shfl_xor(block->c, 0x10);
    block->d = __shfl(block->d, src_thr_d);
    #else
    block->b = __shfl_sync(0xFFFFFFFF, block->b, src_thr_b);
    block->c = __shfl_xor_sync(0xFFFFFFFF, block->c, 0x10);
    block->d = __shfl_sync(0xFFFFFFFF, block->d, src_thr_d);
    #endif
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
    const uint32_t gpuIndex,
    const size_t scratchpadSize,
    const size_t iterations,
    const uint32_t attempt,
    const float intensity)
{
    kernelLaunchParams params;

    size_t free;
    size_t total;

    throw_on_cuda_error(cudaMemGetInfo(&free, &total), __FILE__, __LINE__);

    size_t memoryPerHash = sizeof(block_g) * scratchpadSize;

    /* Make it a multiple of memoryPerHash */
    size_t memoryAvailable = free - (free % memoryPerHash);

    /* Failed to allocate memory - lets try allocating a bit less. */
    if (attempt != 0)
    {
        memoryAvailable = memoryAvailable - (10 * attempt * memoryPerHash);
    }

    /* Cut threads / memory by intensity percentage. Defaults to 100. */
    memoryAvailable = memoryAvailable * (intensity / 100);

    /* The amount of nonces we're going to try per kernel launch */
    params.noncesPerRun = memoryAvailable / memoryPerHash;
    params.noncesPerRun = (params.noncesPerRun / BLAKE_THREADS_PER_BLOCK) * BLAKE_THREADS_PER_BLOCK;

    /* The amount of memory we'll need to allocate on the GPU */
    params.memSize = memoryPerHash * params.noncesPerRun;

    /* Init memory kernel params */
    params.initMemoryBlocks = dim3(params.noncesPerRun / BLAKE_THREADS_PER_BLOCK);
    params.initMemoryThreads = dim3(BLAKE_THREADS_PER_BLOCK, 2);

    params.jobsPerBlock = 16;

    /* Argon2 kernel params */
    params.argon2Blocks = dim3(1, 1, params.noncesPerRun / params.jobsPerBlock);
    params.argon2Threads = dim3(THREADS_PER_LANE, 1, params.jobsPerBlock);
    params.argon2Cache = params.jobsPerBlock * sizeof(u64_shuffle_buf);

    params.getNonceBlocks = params.noncesPerRun / BLAKE_THREADS_PER_BLOCK;
    params.getNonceThreads = BLAKE_THREADS_PER_BLOCK;

    params.scratchpadSize = scratchpadSize;
    params.iterations = iterations;

    return params;
}

/**
 * Stuff we only need to do once (unless the algorithm changes).
 */
NvidiaState initializeState(
    const uint32_t gpuIndex,
    const size_t scratchpadSize,
    const size_t iterations,
    const float intensity,
    uint32_t attempt)
{
    const float nextAttempt = attempt == 0 ? 1 : attempt * 2;

    /* Set current device */
    throw_on_cuda_error(cudaSetDevice(gpuIndex), __FILE__, __LINE__);

    /* Reset, otherwise stream creation will fail */
    throw_on_cuda_error(cudaDeviceReset(), __FILE__, __LINE__);

    /* Don't block CPU execution waiting for kernel to finish */
    throw_on_cuda_error(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), __FILE__, __LINE__);

    NvidiaState state;

    throw_on_cuda_error(cudaStreamCreate(&state.stream), __FILE__, __LINE__);

    state.launchParams = getLaunchParams(gpuIndex, scratchpadSize, iterations, attempt, intensity);

    cudaError_t memoryError = cudaSuccess;

    /* Allocate memory. These things will the be the same size for every job,
       unless the algorithm changes. */
    memoryError = cudaMalloc((void **)&state.memory, state.launchParams.memSize);

    if (memoryError == cudaErrorMemoryAllocation)
    {
        freeState(state);
        return initializeState(gpuIndex, scratchpadSize, iterations, intensity, nextAttempt);
    }

    memoryError = cudaMalloc((void **)&state.nonce, sizeof(uint32_t));

    if (memoryError == cudaErrorMemoryAllocation)
    {
        freeState(state);
        return initializeState(gpuIndex, scratchpadSize, iterations, intensity, nextAttempt);
    }

    memoryError = cudaMalloc((void **)&state.hash, ARGON_HASH_LENGTH);

    if (memoryError == cudaErrorMemoryAllocation)
    {
        freeState(state);
        return initializeState(gpuIndex, scratchpadSize, iterations, intensity, nextAttempt);
    }

    memoryError = cudaMalloc((void **)&state.hashFound, sizeof(bool));

    if (memoryError == cudaErrorMemoryAllocation)
    {
        freeState(state);
        return initializeState(gpuIndex, scratchpadSize, iterations, intensity, nextAttempt);
    }

    memoryError = cudaMalloc((void **)&state.blakeInput, BLAKE_BLOCK_SIZE * 2);

    if (memoryError == cudaErrorMemoryAllocation)
    {
        freeState(state);
        return initializeState(gpuIndex, scratchpadSize, iterations, intensity, nextAttempt);
    }

    throw_on_cuda_error(cudaMemsetAsync(state.hashFound, false, sizeof(bool), state.stream), __FILE__, __LINE__);
    throw_on_cuda_error(cudaMemsetAsync(state.nonce, 0, sizeof(uint32_t), state.stream), __FILE__, __LINE__);

    return state;
}

void freeState(NvidiaState &state)
{
    throw_on_cuda_error(cudaFree(state.memory), __FILE__, __LINE__);
    throw_on_cuda_error(cudaFree(state.nonce), __FILE__, __LINE__);
    throw_on_cuda_error(cudaFree(state.hash), __FILE__, __LINE__);
    throw_on_cuda_error(cudaFree(state.hashFound), __FILE__, __LINE__);
    throw_on_cuda_error(cudaFree(state.blakeInput), __FILE__, __LINE__);

    if (state.stream != NULL)
    {
        throw_on_cuda_error(cudaStreamDestroy(state.stream), __FILE__, __LINE__);
    }
}

void initJob(
    NvidiaState &state,
    const std::vector<uint8_t> &input,
    const std::vector<uint8_t> &saltInput,
    const uint64_t target)
{
    state.target = target;
    setupBlakeInput(input, saltInput, state);
}

HashResult nvidiaHash(NvidiaState &state)
{
    const uint64_t nonceMask = state.isNiceHash ? 0x0000FFFFFF000000UL : 0x00FFFFFFFF000000UL;

    /* Launch the first kernel to perform initial blake initialization */
    initMemoryKernel<<<
        state.launchParams.initMemoryBlocks,
        state.launchParams.initMemoryThreads,
        0, /* No shared memory */
        state.stream
    >>>(
        state.memory,
        state.blakeInput,
        state.blakeInputSize,
        state.localNonce,
        state.launchParams.scratchpadSize,
        nonceMask
    );

    /* Launch the second kernel to perform the main argon work */
    argon2Kernel<<<
        state.launchParams.argon2Blocks,
        state.launchParams.argon2Threads,
        state.launchParams.argon2Cache,
        state.stream
    >>>(
        state.memory,
        state.launchParams.iterations,
        state.launchParams.scratchpadSize / ARGON_SYNC_POINTS
    );

    /* Launch the final kernel to perform final blake round and extract
       nonce that beats the target, if any */
    getNonceKernel<<<
        state.launchParams.getNonceBlocks,
        state.launchParams.getNonceThreads,
        0, /* No shared memory */
        state.stream
    >>>(
        state.memory,
        state.localNonce,
        state.target,
        state.nonce,
        state.hash,
        state.hashFound,
        state.launchParams.scratchpadSize,
        state.isNiceHash,
        state.blakeInput
    );

    /* Wait for kernel */
    throw_on_cuda_error(cudaStreamSynchronize(state.stream), __FILE__, __LINE__);

    HashResult result;

    /* See if we found a valid nonce */
    throw_on_cuda_error(cudaMemcpy(&result.success, state.hashFound, sizeof(result.success), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    if (result.success)
    {
        /* Copy valid nonce + hash back to CPU */
        throw_on_cuda_error(cudaMemcpy(&result.nonce, state.nonce, sizeof(result.nonce), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        throw_on_cuda_error(cudaMemcpy(&result.hash, state.hash, ARGON_HASH_LENGTH, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

        /* Clear the hash found flag so don't think we have found a share when we
           have not, along with the nonce */
        throw_on_cuda_error(cudaMemsetAsync(state.hashFound, false, sizeof(bool), state.stream), __FILE__, __LINE__);
        throw_on_cuda_error(cudaMemsetAsync(state.nonce, 0, sizeof(uint32_t), state.stream), __FILE__, __LINE__);
    }

    return result;
}
