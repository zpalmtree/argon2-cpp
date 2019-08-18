// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

////////////////////
#include "Blake2b.h"
////////////////////

#include <cstring>
#include <limits>
#include <stdexcept>

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

void Blake2b::compressCrossPlatform()
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
        const auto &sigma = SIGMA[i];

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

Blake2b::Blake2b(const Constants::OptimizationMethod optimizationMethod):
    m_hash(8),
    m_chunk(16),
    m_chunkSize(0),
    m_outputHashLength(64),
    m_optimizationMethod(optimizationMethod)
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
