// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

////////////////////
#include "Blake2b.h"
////////////////////

#include <array>
#include <cstring>
#include <stdexcept>

#include <immintrin.h>

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
    return (x >> moves) | (x << (sizeof(T) * 8 - moves));
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

std::vector<uint8_t> Blake2b::Hash(const std::vector<uint8_t> &message)
{
    Blake2b blake;

    blake.Init();
    blake.Update(message);

    return blake.Finalize();
}

std::vector<uint8_t> Blake2b::Hash(const std::string &message)
{
    Blake2b blake;

    blake.Init();
    blake.Update({message.begin(), message.end()});

    return blake.Finalize();
}

Blake2b::Blake2b():
    m_hash(8),
    m_chunk(16),
    m_chunkSize(0),
    m_outputHashLength(64)
{
}

void Blake2b::Init(
    const std::vector<uint8_t> key,
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

    /* Zero the bytes compressed and final block flags */
    std::memset(m_compressXorFlags.data(), 0, 32);

    /* Copy the IV to the hash */
    std::copy(IV.begin(), IV.end(), m_hash.begin());

    /* Mix key size and desired hash length into hash[0] */
    m_hash[0] ^= 0x01010000 ^ (key.size() << 8) ^ outputHashLength;

    if (!key.empty())
    {
        const uint8_t keySize = static_cast<uint8_t>(key.size());
        const uint8_t remainingBytes = 128 - keySize;

        /* Then copy into the next chunk to be processed */
        std::memcpy(&m_chunk[0], &key[0], key.size());

        /* Pad with zeros to make it 128 bytes */
        std::memset(&m_chunk[remainingBytes], 0, remainingBytes);

        /* Signal we have a chunk to process */
        m_chunkSize = 128;

        incrementBytesCompressed(128);
    }
    else
    {
        m_chunkSize = 0;
    }

    
    m_outputHashLength = outputHashLength;
}

/* Break input into 128 byte chunks and process in turn */
void Blake2b::Update(const std::vector<uint8_t> &data)
{
    return Update(&data[0], data.size());
}

void Blake2b::incrementBytesCompressed(const uint64_t bytesCompressed)
{
    /* m_compressXorFlags[0..1] is a 128 bit number stored in little endian. */
    /* Increase the bottom bits */
    m_compressXorFlags[0] += bytesCompressed;

    /* If it's less than the value we just added, we overflowed, and need to
       add one to the top bits */
    m_compressXorFlags[1] += (m_compressXorFlags[0] < bytesCompressed) ? 1 : 0;
}

/* Set all bytes, indicates last block */
void Blake2b::setLastBlock()
{
    m_compressXorFlags[2] = std::numeric_limits<uint64_t>::max();
}

void Blake2b::Update(const uint8_t *data, size_t len)
{
    /* 128 byte chunk to process */
    std::vector<uint64_t> chunk(16);

    size_t offset = 0;

    /* Process 128 bytes at once, aside from final chunk */
    while (len > 0)
    {
        /* Not final block */
        if (m_chunkSize == 128)
        {
            compress();
            m_chunkSize = 0;
        }

        /* Size of chunk to copy */
        uint8_t size = 128 - m_chunkSize;

        if (size > len)
        {
            size = static_cast<uint8_t>(len);
        }

        /* Get void pointer to the chunk vector */
        void *ptr = static_cast<void *>(&m_chunk[0]);

        /* Cast to a uint8_t so we can do math on it */
        /* We need to do the math this way, rather than &m_chunk[m_chunkSize / 8]
           since that does not allow non 8 byte aligned offsets */
        ptr = static_cast<uint8_t *>(ptr) + m_chunkSize;

        std::memcpy(ptr, data + offset, size);

        /* Update stored chunk length */
        m_chunkSize += size;

        /* Update processed byte count */
        incrementBytesCompressed(size);

        len -= size;

        offset += size;
    }
}

std::vector<uint8_t> Blake2b::Finalize()
{
    /* Get void pointer to the chunk vector */
    void *ptr = static_cast<void *>(&m_chunk[0]);

    /* Cast to a uint8_t so we can do math on it */
    /* We need to do the math this way, rather than &m_chunk[m_chunkSize / 8]
       since that does not allow non 8 byte aligned offsets */
    ptr = static_cast<uint8_t *>(ptr) + m_chunkSize;

    /* Pad final chunk with zeros */
    std::memset(ptr, 0, 128 - m_chunkSize);

    /* Set the XOR data for last block */
    setLastBlock();

    /* Process final chunk */
    compress();

    /* Return the final hash as a byte array */
    std::vector<uint8_t> finalHash(m_outputHashLength);

    std::memcpy(&finalHash[0], &m_hash[0], m_outputHashLength);

    return finalHash;
}

__m256i rotr32(__m256i x) {
    return _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));
}

__m256i rotr24(__m256i x) {
    return _mm256_shuffle_epi8(x, _mm256_setr_epi8(
            3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10,
            3, 4, 5, 6, 7, 0, 1, 2, 11, 12, 13, 14, 15, 8, 9, 10
    ));
}

__m256i rotr16(__m256i x) {
    return _mm256_shuffle_epi8(x, _mm256_setr_epi8(
            2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9,
            2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9
    ));
}

__m256i rotr63(__m256i x) {
    return _mm256_xor_si256(_mm256_srli_epi64(x, 63), _mm256_add_epi64(x, x));
}

/*
 * a =  v0,  v1,  v2,  v3
 * b =  v4,  v5,  v6,  v7
 * c =  v8,  v9, v10, v11
 * d = v12, v13, v14, v15
 */
void g1AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]) {
    a = _mm256_add_epi64(a, _mm256_add_epi64(b, _mm256_i32gather_epi64((long long int*)blk, vindex[r][0], 8)));
    d = rotr32(_mm256_xor_si256(a, d));
    c = _mm256_add_epi64(c, d);
    b = rotr24(_mm256_xor_si256(b, c));

    a = _mm256_add_epi64(a, _mm256_add_epi64(b, _mm256_i32gather_epi64((long long int*)blk, vindex[r][1], 8)));
    d = rotr16(_mm256_xor_si256(a, d));
    c = _mm256_add_epi64(c, d);
    b = rotr63(_mm256_xor_si256(b, c));
}

void g2AVX2(uint32_t r, __m256i& a, __m256i& b, __m256i& c, __m256i& d, uint64_t* blk, const __m128i vindex[12][4]) {
    a = _mm256_add_epi64(a, _mm256_add_epi64(b, _mm256_i32gather_epi64((long long int*)blk, vindex[r][2], 8)));
    d = rotr32(_mm256_xor_si256(a, d));
    c = _mm256_add_epi64(c, d);
    b = rotr24(_mm256_xor_si256(b, c));

    a = _mm256_add_epi64(a, _mm256_add_epi64(b, _mm256_i32gather_epi64((long long int*)blk, vindex[r][3], 8)));
    d = rotr16(_mm256_xor_si256(a, d));
    c = _mm256_add_epi64(c, d);
    b = rotr63(_mm256_xor_si256(b, c));
}

void diagonalize(__m256i& b, __m256i& c, __m256i& d) {
    b = _mm256_permute4x64_epi64(b, _MM_SHUFFLE(0, 3, 2, 1));
    c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(1, 0, 3, 2));
    d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(2, 1, 0, 3));
}

void undiagonalize(__m256i& b, __m256i& c, __m256i& d) {
    b = _mm256_permute4x64_epi64(b, _MM_SHUFFLE(2, 1, 0, 3));
    c = _mm256_permute4x64_epi64(c, _MM_SHUFFLE(1, 0, 3, 2));
    d = _mm256_permute4x64_epi64(d, _MM_SHUFFLE(0, 3, 2, 1));
}

void Blake2b::compressAVX2()
{
    static const __m128i vindex[12][4] = {
        { _mm_set_epi32( 6,  4,  2,  0), _mm_set_epi32( 7,  5,  3,  1), _mm_set_epi32(14, 12, 10,  8), _mm_set_epi32(15, 13, 11,  9) },
        { _mm_set_epi32(13,  9,  4, 14), _mm_set_epi32( 6, 15,  8, 10), _mm_set_epi32( 5, 11,  0,  1), _mm_set_epi32( 3,  7,  2, 12) },
        { _mm_set_epi32(15,  5, 12, 11), _mm_set_epi32(13,  2,  0,  8), _mm_set_epi32( 9,  7,  3, 10), _mm_set_epi32( 4,  1,  6, 14) },
        { _mm_set_epi32(11, 13,  3,  7), _mm_set_epi32(14, 12,  1,  9), _mm_set_epi32(15,  4,  5,  2), _mm_set_epi32( 8,  0, 10,  6) },
        { _mm_set_epi32(10,  2,  5,  9), _mm_set_epi32(15,  4,  7,  0), _mm_set_epi32( 3,  6, 11, 14), _mm_set_epi32(13,  8, 12,  1) },
        { _mm_set_epi32( 8,  0,  6,  2), _mm_set_epi32( 3, 11, 10, 12), _mm_set_epi32( 1, 15,  7,  4), _mm_set_epi32( 9, 14,  5, 13) },
        { _mm_set_epi32( 4, 14,  1, 12), _mm_set_epi32(10, 13, 15,  5), _mm_set_epi32( 8,  9,  6,  0), _mm_set_epi32(11,  2,  3,  7) },
        { _mm_set_epi32( 3, 12,  7, 13), _mm_set_epi32( 9,  1, 14, 11), _mm_set_epi32( 2,  8, 15,  5), _mm_set_epi32(10,  6,  4,  0) },
        { _mm_set_epi32( 0, 11, 14,  6), _mm_set_epi32( 8,  3,  9, 15), _mm_set_epi32(10,  1, 13, 12), _mm_set_epi32( 5,  4,  7,  2) },
        { _mm_set_epi32( 1,  7,  8, 10), _mm_set_epi32( 5,  6,  4,  2), _mm_set_epi32(13,  3,  9, 15), _mm_set_epi32( 0, 12, 14, 11) },
        { _mm_set_epi32( 6,  4,  2,  0), _mm_set_epi32( 7,  5,  3,  1), _mm_set_epi32(14, 12, 10,  8), _mm_set_epi32(15, 13, 11,  9) },
        { _mm_set_epi32(13,  9,  4, 14), _mm_set_epi32( 6, 15,  8, 10), _mm_set_epi32( 5, 11,  0,  1), _mm_set_epi32( 3,  7,  2, 12) },
    };

    static const __m256i iv[2] = {
        _mm256_set_epi64x(0xa54ff53a5f1d36f1ULL, 0x3c6ef372fe94f82bULL, 0xbb67ae8584caa73bULL, 0x6a09e667f3bcc908ULL),
        _mm256_set_epi64x(0x5be0cd19137e2179ULL, 0x1f83d9abfb41bd6bULL, 0x9b05688c2b3e6c1fULL, 0x510e527fade682d1ULL)
    };

    __m256i a = _mm256_loadu_si256((__m256i*)&m_hash[0]);
    __m256i b = _mm256_loadu_si256((__m256i*)&m_hash[4]);
    __m256i c = iv[0];
    __m256i d = _mm256_xor_si256(iv[1], _mm256_loadu_si256((__m256i*)&m_compressXorFlags[0]));

    for(uint32_t i = 0; i < 12; i++)
    {
        g1AVX2(i, a, b, c, d, m_chunk.data(), vindex);
        diagonalize(b, c, d);
        g2AVX2(i, a, b, c, d, m_chunk.data(), vindex);
        undiagonalize(b, c, d);
    }

    _mm256_storeu_si256((__m256i*)&m_hash[0], _mm256_xor_si256(
            _mm256_loadu_si256((__m256i*)&m_hash[0]),
            _mm256_xor_si256(a, c)
    ));
    _mm256_storeu_si256(((__m256i*)&m_hash[0]) + 1, _mm256_xor_si256(
            _mm256_loadu_si256(((__m256i*)&m_hash[0]) + 1),
            _mm256_xor_si256(b, d)
    ));
}

void Blake2b::compress()
{
    std::vector<uint64_t> v(16);

    /* v[0..7] = h[0..7] */
    std::copy(m_hash.begin(), m_hash.end(), v.begin());

    /* v[8..15] = IV[0..7] */
    std::copy(IV.begin(), IV.end(), v.begin() + 8);

    v[12] ^= m_compressXorFlags[0];
    v[13] ^= m_compressXorFlags[1];
    v[14] ^= m_compressXorFlags[2];
    v[15] ^= m_compressXorFlags[3];
 
    /* 12 rounds of mixing */
    for (int i = 0; i < 12; i++)
    {
        /* Get the sigma constant for the current round */
        const auto &sigma = SIGMA[i % 10];

        /* Column round */
        mix(v[0], v[4], v[8],  v[12], m_chunk[sigma[0]],  m_chunk[sigma[1]]);
        mix(v[1], v[5], v[9],  v[13], m_chunk[sigma[2]],  m_chunk[sigma[3]]);
        mix(v[2], v[6], v[10], v[14], m_chunk[sigma[4]],  m_chunk[sigma[5]]);
        mix(v[3], v[7], v[11], v[15], m_chunk[sigma[6]],  m_chunk[sigma[7]]);

        /* Diagonal round */
        mix(v[0], v[5], v[10], v[15], m_chunk[sigma[8]],  m_chunk[sigma[9]]);
        mix(v[1], v[6], v[11], v[12], m_chunk[sigma[10]], m_chunk[sigma[11]]);
        mix(v[2], v[7], v[8],  v[13], m_chunk[sigma[12]], m_chunk[sigma[13]]);
        mix(v[3], v[4], v[9],  v[14], m_chunk[sigma[14]], m_chunk[sigma[15]]);
    }

    for (int i = 0; i < 8; i++)
    {
        m_hash[i] ^= v[i] ^ v[i + 8];
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
