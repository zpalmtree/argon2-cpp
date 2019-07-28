#include <iostream>

#include <vector>

#include "Argon2/Argon2.h"

#include "Blake2/Blake2b.h"

#include "Utilities/Utilities.h"

int main()
{
    if (byteArrayToHexString(Blake2b::Hash("")) != "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce")
    {
        throw std::runtime_error("Bad result");
    }

    if (byteArrayToHexString(Blake2b::Hash("The quick brown fox jumps over the lazy dog")) != "a8add4bdddfd93e4877d2746e62817b116364a1fa7bc148d95090bc7333b3673f82401cf7aa2e4cb1ecd90296e3f14cb5413f8ed77be73045b13914cdcd6a918")
    {
        throw std::runtime_error("Bad result");
    }

    const std::vector<uint8_t> password = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    };

    const std::vector<uint8_t> salt = {
        2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2
    };

    const std::vector<uint8_t> key = {
        3, 3, 3, 3, 3, 3, 3, 3
    };

    const std::vector<uint8_t> associatedData = {
        4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4
    };

    const auto argon2D = deriveKey(
        0,
        password,
        salt,
        key,
        associatedData,
        3,
        32,
        4,
        32
    );

    std::cout << byteArrayToHexString(argon2D) << std::endl;

    const auto argon2I = deriveKey(
        1,
        password,
        salt,
        key,
        associatedData,
        3,
        32,
        4,
        32
    );

    std::cout << byteArrayToHexString(argon2I) << std::endl;

    const auto argon2ID = deriveKey(
        2,
        password,
        salt,
        key,
        associatedData,
        3,
        32,
        4,
        32
    );

    std::cout << byteArrayToHexString(argon2ID) << std::endl;

    std::cout << "Tests passed" << std::endl;

    return 0;
}
