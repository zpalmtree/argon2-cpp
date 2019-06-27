// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <cstdint>

#include <string>

#include <vector>

std::vector<uint8_t> Blake2b(const std::vector<uint8_t> &message);

std::vector<uint8_t> Blake2b(const std::string &message);
