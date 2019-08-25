/*
    BLAKE2 AVX2 source code package
    Copyright 2016, Samuel Neves <sneves@dei.uc.pt>.

    2019 Creative Commons Legal Code

    CC0 1.0 Universal

    CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
    LEGAL SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN
    ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
    INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
    REGARDING THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS
    PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM
    THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
    HEREUNDER.
*/

// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

////////////////////////////////////////
#include "Intrinsics/X86/CompressAVX2.h"
////////////////////////////////////////

#include "Intrinsics/X86/LoadAVX2.h"
#include "Intrinsics/X86/RotationsAVX2.h"

namespace CompressAVX2
{
    void g1AVX2(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& m)
    {
        a = _mm256_add_epi64(_mm256_add_epi64(a, m), b);
        d = RotationsAVX2::rotr32(_mm256_xor_si256(d, a));
        c = _mm256_add_epi64(c, d);
        b = RotationsAVX2::rotr24(_mm256_xor_si256(b, c));
    }

    void g2AVX2(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& m)
    {
        a = _mm256_add_epi64(_mm256_add_epi64(a, m), b);
        d = RotationsAVX2::rotr16(_mm256_xor_si256(d, a));
        c = _mm256_add_epi64(c, d);
        b = RotationsAVX2::rotr63(_mm256_xor_si256(b, c));
    }

    void diagonalizeAVX2(__m256i& a, __m256i& c, __m256i& d)
    {
        a = _mm256_permute4x64_epi64(a, _MM_SHUFFLE(2, 1, 0, 3));
        d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(1, 0, 3, 2));
        c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(0, 3, 2, 1));
    }

    void undiagonalizeAVX2(__m256i& a, __m256i& c, __m256i& d)
    {
        a = _mm256_permute4x64_epi64(a, _MM_SHUFFLE(0, 3, 2, 1));
        d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(1, 0, 3, 2));
        c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(2, 1, 0, 3));
    }

    #define BLAKE2B_ROUND_V1(a, b, c, d, r)               \
        do                                                \
        {                                                 \
            __m256i b0;                                   \
            BLAKE2B_LOAD_MSG_ ##r ##_1(b0);               \
            g1AVX2(a, b, c, d, b0);                       \
                                                          \
            BLAKE2B_LOAD_MSG_ ##r ##_2(b0);               \
            g2AVX2(a, b, c, d, b0);                       \
                                                          \
            diagonalizeAVX2(a, c, d);                     \
            BLAKE2B_LOAD_MSG_ ##r ##_3(b0);               \
                                                          \
            g1AVX2(a, b, c, d, b0);                       \
            BLAKE2B_LOAD_MSG_ ##r ##_4(b0);               \
                                                          \
            g2AVX2(a, b, c, d, b0);                       \
            undiagonalizeAVX2(a, c, d);                   \
        } while(0)

    void compressAVX2(
        std::vector<uint64_t> &hash,
        std::vector<uint64_t> &chunk,
        std::vector<uint64_t> &compressXorFlags)
    {
        __m256i m0;
        __m256i m1;
        __m256i m2;
        __m256i m3;
        __m256i m4;
        __m256i m5;
        __m256i m6;
        __m256i m7;

        #if _MSC_VER && !__INTEL_COMPILER
        /* MSVC messes up the loading with _mm256_broadcastsi128_si256.
           We disable optimizations to fix this for MSVC. To avoid deoptimizing
           the whole compress function, we split it into a small function to
           only take effect there */
        loadChunk(m0, m1, m2, m3, m4, m5, m6, m7, chunk);
        #else
        m0 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[0])));
        m1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[2])));
        m2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[4])));
        m3 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[6])));
        m4 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[8])));
        m5 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[10])));
        m6 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[12])));
        m7 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&chunk[14])));
        #endif

        static const __m256i iv[2] = {
            _mm256_set_epi64x(0xa54ff53a5f1d36f1ULL, 0x3c6ef372fe94f82bULL, 0xbb67ae8584caa73bULL, 0x6a09e667f3bcc908ULL),
            _mm256_set_epi64x(0x5be0cd19137e2179ULL, 0x1f83d9abfb41bd6bULL, 0x9b05688c2b3e6c1fULL, 0x510e527fade682d1ULL)
        };

        const __m256i iv0 = _mm256_loadu_si256((__m256i*)&hash[0]);
        const __m256i iv1 = _mm256_loadu_si256((__m256i*)&hash[4]);

        __m256i a = iv0;
        __m256i b = iv1;

        __m256i t0, t1;

        __m256i c = iv[0];
        __m256i d = _mm256_xor_si256(iv[1], _mm256_loadu_si256((__m256i*)&compressXorFlags[0]));

        BLAKE2B_ROUND_V1(a, b, c, d,  0);
        BLAKE2B_ROUND_V1(a, b, c, d,  1);
        BLAKE2B_ROUND_V1(a, b, c, d,  2);
        BLAKE2B_ROUND_V1(a, b, c, d,  3);
        BLAKE2B_ROUND_V1(a, b, c, d,  4);
        BLAKE2B_ROUND_V1(a, b, c, d,  5);
        BLAKE2B_ROUND_V1(a, b, c, d,  6);
        BLAKE2B_ROUND_V1(a, b, c, d,  7);
        BLAKE2B_ROUND_V1(a, b, c, d,  8);
        BLAKE2B_ROUND_V1(a, b, c, d,  9);
        BLAKE2B_ROUND_V1(a, b, c, d, 10);
        BLAKE2B_ROUND_V1(a, b, c, d, 11);

        _mm256_storeu_si256((__m256i*)&hash[0], _mm256_xor_si256(iv0, _mm256_xor_si256(a, c)));
        _mm256_storeu_si256((__m256i*)&hash[4], _mm256_xor_si256(iv1, _mm256_xor_si256(b, d)));
    }
}
