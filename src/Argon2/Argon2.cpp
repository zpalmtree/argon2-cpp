// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

///////////////////
#include "Argon2.h"
///////////////////

#include "Blake2/Blake2b.h"

#include <cstring>

#include <stdexcept>

constexpr uint32_t CURRENT_ARGON_VERSION = 19;

std::vector<uint8_t> Argon2d(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations)
{
    return Argon2Internal(
        message,
        salt,
        parallelism,
        outputHashLength,
        memoryUsageKB,
        iterations,
        CURRENT_ARGON_VERSION,
        {},
        {},
        0
    );
}

std::vector<uint8_t> Argon2i(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations)
{
    return Argon2Internal(
        message,
        salt,
        parallelism,
        outputHashLength,
        memoryUsageKB,
        iterations,
        CURRENT_ARGON_VERSION,
        {},
        {},
        1
    );
}

std::vector<uint8_t> Argon2id(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations)
{
    return Argon2Internal(
        message,
        salt,
        parallelism,
        outputHashLength,
        memoryUsageKB,
        iterations,
        CURRENT_ARGON_VERSION,
        {},
        {},
        2
    );
}

std::vector<uint8_t> Argon2Internal(
    const std::vector<uint8_t> &message,
    const std::vector<uint8_t> &salt,
    const uint32_t parallelism,
    const uint32_t outputHashLength,
    const uint32_t memoryUsageKB,
    const uint32_t iterations,
    const uint32_t version,
    const std::vector<uint8_t> key,
    const std::vector<uint8_t> associatedData,
    const uint32_t hashType)
{
    /* STEP 1: Validate parameters are good */
    if (salt.size() < 8)
    {
        throw std::invalid_argument("Salt must be at least 8 bytes!");
    }

    if (parallelism == 0 || parallelism > 16777215)
    {
        throw std::invalid_argument("Parallelism must be between 1 and 2^24 - 1!");
    }

    if (outputHashLength < 4)
    {
        throw std::invalid_argument("Output hash length must be at least 4 bytes!");
    }

    if (memoryUsageKB < 8 * parallelism)
    {
        throw std::invalid_argument("Memory usage must be at least 8 * parallelism (kb)!");
    }

    if (iterations == 0)
    {
        throw std::invalid_argument("Iterations must be at least 1!");
    }

    if (version != CURRENT_ARGON_VERSION)
    {
        throw std::invalid_argument("Version must be equal to " + std::to_string(CURRENT_ARGON_VERSION) + "!");
    }

    if (hashType != 0 && hashType != 1 && hashType != 2)
    {
        throw std::invalid_argument("Hash type must be 0, 1, or 2!");
    }

    uint32_t size;

    /* Can either do this step by appending everything to one vector then
       performing one hash, or using a streaming approach with .Update() */
    class Blake2b blake;

    blake.Init();

    blake.Update((const uint8_t *)&parallelism, sizeof(parallelism));
    blake.Update((const uint8_t *)&outputHashLength, sizeof(outputHashLength));
    blake.Update((const uint8_t *)&memoryUsageKB, sizeof(memoryUsageKB));
    blake.Update((const uint8_t *)&iterations, sizeof(iterations));
    blake.Update((const uint8_t *)&version, sizeof(version));
    blake.Update((const uint8_t *)&hashType, sizeof(hashType));

    size = message.size();
    blake.Update((const uint8_t *)&size, sizeof(size));
    blake.Update(&message[0], size);

    size = salt.size();
    blake.Update((const uint8_t *)&size, sizeof(size));
    blake.Update(&salt[0], size);

    size = key.size();
    blake.Update((const uint8_t *)&size, sizeof(size));
    blake.Update(&key[0], size);

    size = associatedData.size();
    blake.Update((const uint8_t *)&size, sizeof(size));
    blake.Update(&associatedData[0], size);

    return blake.Finalize();
}
