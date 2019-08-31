// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

#include <vector>

std::vector<uint8_t> nvidiaHash(const std::vector<uint8_t> &message, const std::vector<uint8_t> &saltInput);
