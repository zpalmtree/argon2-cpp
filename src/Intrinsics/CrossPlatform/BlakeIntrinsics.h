// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

inline void Blake2b::compress()
{
    compressCrossPlatform();
}

/* Just to avoid possible linking errors */
inline void Blake2b::compressAVX2()
{
}
