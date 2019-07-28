// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

/////////////////////
#include "Utilities.h"
/////////////////////

#include <sstream>

#include <iomanip>

std::string byteArrayToHexString(const std::vector<uint8_t> &input)
{
    std::stringstream ss;
    ss << std::hex << std::setfill('0');

    for (const auto c : input)
    {
        ss << std::setw(2) << static_cast<unsigned>(c);
    }

    return ss.str();
}
