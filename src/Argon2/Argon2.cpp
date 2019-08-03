// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

///////////////////
#include "Argon2.h"
///////////////////

#include "Argon2/Constants.h"

#include "Blake2/Blake2b.h"

#include "Utilities/Utilities.h"

#include <cmath>

#include <cstring>

#include <stdexcept>

#include <sstream>

#include <tuple>

std::vector<uint8_t> Argon2d(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t time, /* Or iterations */
    const uint32_t memory,
    const uint32_t threads, /* Or parallelism */
    const uint32_t keyLen /* Output hash length */)
{
    return deriveKey(
        Constants::ARGON2D,
        message,
        salt,
        {},
        {},
        time,
        memory,
        threads,
        keyLen
    );
}

std::vector<uint8_t> Argon2i(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t time, /* Or iterations */
    const uint32_t memory,
    const uint32_t threads, /* Or parallelism */
    const uint32_t keyLen /* Output hash length */)
{
    return deriveKey(
        Constants::ARGON2I,
        message,
        salt,
        {},
        {},
        time,
        memory,
        threads,
        keyLen
    );
}

std::vector<uint8_t> Argon2id(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t time, /* Or iterations */
    const uint32_t memory,
    const uint32_t threads, /* Or parallelism */
    const uint32_t keyLen /* Output hash length */)
{
    return deriveKey(
        Constants::ARGON2ID,
        message,
        salt,
        {},
        {},
        time,
        memory,
        threads,
        keyLen
    );
}

std::vector<uint8_t> deriveKey(
    const uint32_t mode,
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const std::vector<uint8_t> &secret,
    const std::vector<uint8_t> &data,
    const uint32_t time,
    uint32_t memory,
    const uint32_t threads,
    const uint32_t keyLen)
{
    validateParameters(salt, threads, keyLen, memory, time, mode);

    std::vector<uint8_t> h0 = initHash(
        message, salt, secret, data, time, memory, threads, keyLen, mode
    );

    memory = memory / (Constants::SYNC_POINTS * threads) * (Constants::SYNC_POINTS * threads);

    if (memory < 2 * Constants::SYNC_POINTS * threads)
    {
        memory = 2 * Constants::SYNC_POINTS * threads;
    }

    auto B = initBlocks(h0, memory, threads);

    processBlocks(B, time, memory, threads, mode);

    return extractKey(B, memory, threads, keyLen);
}

std::vector<uint8_t> initHash(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const std::vector<uint8_t> &key,
    const std::vector<uint8_t> &data,
    const uint32_t time,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t keyLen,
    const uint32_t mode)
{
    const uint32_t version = Constants::CURRENT_ARGON_VERSION;

    uint32_t size;

    /* Can either do this step by appending everything to one vector then
       performing one hash, or using a streaming approach with .Update() */
    Blake2b blake;

    /* STEP 2: Initialize first hash */
    blake.Init();

    blake.Update(reinterpret_cast<const uint8_t *>(&threads), sizeof(threads));
    blake.Update(reinterpret_cast<const uint8_t *>(&keyLen), sizeof(keyLen));
    blake.Update(reinterpret_cast<const uint8_t *>(&memory), sizeof(memory));
    blake.Update(reinterpret_cast<const uint8_t *>(&time), sizeof(time));
    blake.Update(reinterpret_cast<const uint8_t *>(&version), sizeof(version));
    blake.Update(reinterpret_cast<const uint8_t *>(&mode), sizeof(mode));

    size = message.size();
    blake.Update(reinterpret_cast<const uint8_t *>(&size), sizeof(size));
    blake.Update(message.data(), size);

    size = salt.size();
    blake.Update(reinterpret_cast<const uint8_t *>(&size), sizeof(size));
    blake.Update(salt.data(), size);

    size = key.size();
    blake.Update(reinterpret_cast<const uint8_t *>(&size), sizeof(size));
    blake.Update(key.data(), size);

    size = data.size();
    blake.Update(reinterpret_cast<const uint8_t *>(&size), sizeof(size));
    blake.Update(data.data(), size);

    return blake.Finalize();
}

std::vector<Block> initBlocks(
    std::vector<uint8_t> &h0,
    const uint32_t memory,
    const uint32_t threads)
{
    h0.resize(Constants::INITIAL_HASH_SIZE, 0);

    uint8_t block0[Constants::BLOCK_SIZE_BYTES];

    std::vector<Block> B(memory);

    for (uint32_t lane = 0; lane < threads; lane++)
    {
        int j = lane * (memory / threads);

        /* Pop 0 into h0[64..67] */
        h0[64] = 0;

        /* Copy lane into h0[68..71] */
        std::memcpy(&h0[64 + 4], &lane, sizeof(uint32_t));

        blake2bHash(block0, h0, Constants::BLOCK_SIZE_BYTES);

        for (int i = 0; i < Constants::BLOCK_SIZE; i++)
        {
            std::memcpy(&B[j][i], &block0[i * 8], sizeof(uint64_t));
        }

        /* Pop 1 into hash[64..67] */
        h0[64] = 1;

        blake2bHash(block0, h0, Constants::BLOCK_SIZE_BYTES);

        for (int i = 0; i < Constants::BLOCK_SIZE; i++)
        {
            std::vector<uint8_t> tmp;

            tmp.assign(&block0[i * 8], &block0[8 + (i * 8)]);
            std::memcpy(&B[j+1][i], &block0[i * 8], sizeof(uint64_t));
        }
    }

    return B;
}

void processBlocks(
    std::vector<Block> &B,
    const uint32_t time,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t mode)
{
    for (uint32_t i = 0; i < time; i++)
    {
        for (uint32_t slice = 0; slice < Constants::SYNC_POINTS; slice++)
        {
            for (uint32_t lane = 0; lane < threads; lane++)
            {
                /* TODO: thread */
                /* Maybe use a std::function<> lambda? */
                processSegment(i, slice, lane, B, mode, memory, time, threads);
            }
        }
    }
}

void processSegment(
    const uint32_t n,
    const uint32_t slice,
    const uint32_t lane,
    std::vector<Block> &B,
    const uint32_t mode,
    const uint32_t memory,
    const uint32_t time,
    const uint32_t threads)
{
    const uint32_t lanes = memory / threads;
    const uint32_t segments = lanes / Constants::SYNC_POINTS;

    /* Default initializing to zero */
    Block addresses {};
    Block in {};
    Block zero {};

    if (mode == Constants::ARGON2I
    || (mode == Constants::ARGON2ID && n == 0 && slice < Constants::SYNC_POINTS / 2))
    {
        in[0] = n;
        in[1] = lane;
        in[2] = slice;
        in[3] = memory;
        in[4] = time;
        in[5] = mode;
    }

    uint32_t index = 0;

    if (n == 0 && slice == 0)
    {
        index = 2;

        if (mode == Constants::ARGON2I || mode == Constants::ARGON2ID)
        {
            in[6]++;
            processBlock(addresses, in, zero);
            processBlock(addresses, addresses, zero);
        }
    }

    uint32_t offset = lane * lanes + slice * segments + index;

    uint64_t random;

    while (index < segments)
    {
        uint32_t prev = offset - 1;

        /* Last block in lane */
        if (index == 0 && slice == 0)
        {
            prev += lanes;
        }

        /* TODO: Combine */
        if (mode == Constants::ARGON2I
        || (mode == Constants::ARGON2ID && n == 0 && slice < Constants::SYNC_POINTS / 2))
        {
            if (index % Constants::BLOCK_SIZE == 0)
            {
                in[6]++;
                processBlock(addresses, in, zero);
                processBlock(addresses, addresses, zero);
            }

            random = addresses[index % Constants::BLOCK_SIZE];
        }
        else
        {
            random = B[prev][0];
        }

        uint32_t newOffset = indexAlpha(random, lanes, segments, threads, n, slice, lane, index);

        processBlockXOR(B[offset], B[prev], B[newOffset]);

        index++;
        offset++;
    }
}

void blake2bHash(
    uint8_t *out,
    std::vector<uint8_t> input,
    uint32_t outputLength)
{
    /* Prepend the length of the output hash length to the input data */
    input.insert(
        input.begin(),
        reinterpret_cast<const uint8_t *>(&outputLength),
        reinterpret_cast<const uint8_t *>(&outputLength + 1)
    );

    Blake2b blake;

    if (outputLength < Constants::HASH_SIZE)
    {
        blake.Init({}, outputLength);
    }
    else
    {
        blake.Init();
    }

    blake.Update(input.data(), input.size());

    std::vector<uint8_t> buffer = blake.Finalize();

    if (outputLength < Constants::HASH_SIZE)
    {
        std::copy(buffer.begin(), buffer.end(), out);
        return;
    }

    blake.Init();

    std::copy(buffer.begin(), buffer.begin() + 32, out);

    out += 32;
    outputLength -= 32;

    while (outputLength > Constants::HASH_SIZE)
    {
        /* TODO: Use blake directly? */
        blake.Update(buffer.data(), buffer.size());
        buffer = blake.Finalize();

        std::copy(buffer.begin(), buffer.begin() + 32, out);

        out += 32;
        outputLength -= 32;

        blake.Init();
    }

    if (outputLength % Constants::HASH_SIZE > 0)
    {
        uint32_t r = ((outputLength + 31) / 32) - 2;
        blake.Init({}, outputLength - 32 * r);
    }

    blake.Update(buffer.data(), buffer.size());

    buffer = blake.Finalize();

    std::copy(buffer.begin(), buffer.end(), out);
}

std::vector<uint8_t> extractKey(
    std::vector<Block> &B,
    const uint32_t memory,
    const uint32_t threads,
    const uint32_t keyLen)
{
    const uint32_t lanes = memory / threads;

    for (uint32_t lane = 0; lane < threads - 1; lane++)
    {
        for (uint32_t i = 0; i < Constants::BLOCK_SIZE; i++)
        {
            B[memory - 1][i] ^= B[(lane * lanes) + lanes - 1][i];
        }
    }

    std::vector<uint8_t> block(Constants::BLOCK_SIZE_BYTES);

    for (uint32_t i = 0; i < Constants::BLOCK_SIZE; i++)
    {
        std::memcpy(&block[i * 8], &B[memory - 1][i], sizeof(uint64_t));
    }

    std::vector<uint8_t> key(keyLen);

    blake2bHash(key.data(), block, keyLen);

    return key;
}

void processBlockGeneric(
    Block &out,
    Block &in1,
    Block &in2,
    const bool doXor)
{
    Block t;

    for (int i = 0; i < Constants::BLOCK_SIZE; i++)
    {
        t[i] = in1[i] ^ in2[i];
    }

    for (int i = 0; i < Constants::BLOCK_SIZE; i += 16)
    {
        blamkaGeneric(
            t[i + 0],
            t[i + 1],
            t[i + 2],
            t[i + 3],
            t[i + 4],
            t[i + 5],
            t[i + 6],
            t[i + 7],
            t[i + 8],
            t[i + 9],
            t[i + 10],
            t[i + 11],
            t[i + 12],
            t[i + 13],
            t[i + 14],
            t[i + 15]
        );
    }

    for (int i = 0; i < Constants::BLOCK_SIZE / 8; i += 2)
    {
        blamkaGeneric(
            t[0 + i + 0],
            t[0 + i + 1],
            t[16 + i + 0],
            t[16 + i + 1],
            t[32 + i + 0],
            t[32 + i + 1],
            t[48 + i + 0],
            t[48 + i + 1],
            t[64 + i + 0],
            t[64 + i + 1],
            t[80 + i + 0],
            t[80 + i + 1],
            t[96 + i + 0],
            t[96 + i + 1],
            t[112 + i + 0],
            t[112 + i + 1]
        );
    }

    if (doXor)
    {
        for (int i = 0; i < Constants::BLOCK_SIZE; i++)
        {
            out[i] ^= in1[i] ^ in2[i] ^ t[i];
        }
    }
    else
    {
        for (int i = 0; i < Constants::BLOCK_SIZE; i++)
        {
            out[i] = in1[i] ^ in2[i] ^ t[i];
        }
    }
}

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

void processBlock(
    Block &out,
    Block &in1,
    Block &in2)
{
    processBlockGeneric(out, in1, in2, false);
}

void processBlockXOR(
    Block &out,
    Block &in1,
    Block &in2)
{
    processBlockGeneric(out, in1, in2, true);
}

uint32_t indexAlpha(
    const uint64_t random,
    const uint32_t lanes,
    const uint32_t segments,
    const uint32_t threads,
    const uint32_t n,
    const uint32_t slice,
    const uint32_t lane,
    const uint32_t index)
{
    uint32_t refLane = static_cast<uint32_t>(random >> 32) % threads;

    if (n == 0 && slice == 0)
    {
        refLane = lane;
    }

    uint32_t m = 3 * segments;
    uint32_t s = ((slice + 1) % Constants::SYNC_POINTS) * segments;

    if (lane == refLane)
    {
        m += index;
    }

    if (n == 0)
    {
        m = slice * segments;
        s = 0;

        if (slice == 0 || lane == refLane)
        {
            m += index;
        }
    }

    if (index == 0 || lane == refLane)
    {
        m--;
    }

    return phi(random, m, s, refLane, lanes);
}

uint32_t phi(
    const uint64_t random,
    uint64_t m,
    uint64_t s,
    const uint32_t lane,
    const uint32_t lanes)
{
    uint64_t p = random & 0xFFFFFFFF;
    p = (p * p) >> 32;
    p = (p * m) >> 32;

    return lane * lanes + static_cast<uint32_t>((s + m - (p + 1)) % static_cast<uint64_t>(lanes));
}

void validateParameters(
    const std::vector<uint8_t> &salt,
    const uint32_t threads,
    const uint32_t keyLen,
    const uint32_t memory,
    const uint32_t time,
    const uint32_t mode)
{

    if (salt.size() < Constants::MIN_SALT_SIZE)
    {
        throw std::invalid_argument("Salt must be at least 8 bytes!");
    }

    if (threads == 0 || threads > Constants::MAX_PARALLELISM)
    {
        throw std::invalid_argument("Threads must be between 1 and 2^24 - 1!");
    }

    if (keyLen < Constants::MIN_OUTPUT_HASH_LENGTH)
    {
        throw std::invalid_argument("Key len must be at least 4 bytes!");
    }

    if (memory < Constants::MIN_PARALLELISM_FACTOR * threads)
    {
        throw std::invalid_argument("Memory must be at least 8 * threads (kb)!");
    }

    if (time == 0)
    {
        throw std::invalid_argument("Time must be at least 1!");
    }

    if (mode != Constants::ARGON2D
     && mode != Constants::ARGON2I
     && mode != Constants::ARGON2ID)
    {
        std::stringstream stream;

        stream << "Mode must be " << Constants::ARGON2D << ", "
               << Constants::ARGON2I << ", or " << Constants::ARGON2ID << "!";

        throw std::invalid_argument(stream.str());
    }
}
