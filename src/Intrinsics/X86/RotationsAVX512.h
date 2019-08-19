// Copyright (c) 2019, Zpalmtree
//
// Please see the included LICENSE file for more information.

#pragma once

namespace RotationsAVX512
{
    inline __m512i rotr16(__m512i x)
    {
        return _mm512_ror_epi64(x, 16);
    }

    inline __m512i rotr24(__m512i x)
    {
        return _mm512_ror_epi64(x, 24);
    }

    inline __m512i rotr32(__m512i x)
    {
        return _mm512_ror_epi64(x, 32);
    }

    inline __m512i rotr63(__m512i x)
    {
        return _mm512_ror_epi64(x, 63);
    }
}
