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
