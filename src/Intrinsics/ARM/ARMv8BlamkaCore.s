
	// expects a block in v0...v7 and returns the result in v0...v7

	"mov v1.d[0], v0.d[1]\n"
	"mov v3.d[0], v4.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v0.2s, v4.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v4.2d, v3.2d\n" 
	"add v0.2d, v0.2d, v1.2d\n" 
	// (v00,v01) = (v04,v05) + 2 * LOWER(v00,v01) * LOWER (v04,v05)

	"eor v12.16b, v12.16b, v0.16b\n"
	// (v12,v13) ^= (v00,v1)
	"shl v1.2d, v12.2d, #32\n"
	"ushr v3.2d, v12.2d, #32\n"
	"orr v12.16b, v3.16b, v1.16b\n"
	//  (v12,v13) = (v12,v13)>>32 | (v12,v13)<<32;

	"mov v1.d[0], v8.d[1]\n"
	"mov v3.d[0], v12.d[1]\n" 
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v8.2s, v12.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v12.2d, v3.2d\n" 
	"add v8.2d, v8.2d, v1.2d\n" 
	// (v08,v09) = (v12,v13) + 2 * LOWER(v08,v12) * LOWER (v08,v13)

	"eor v4.16b, v4.16b, v8.16b\n"
	// (v04,v05) ^= (v08,v08)
	"shl v1.2d, v4.2d, #40\n"
	"ushr v3.2d, v4.2d, #24\n"
	"orr v4.16b, v3.16b, v1.16b\n"
	// (v04,v05) = (v04,v05)>>24 | (v04,v05)<<40

	"mov v1.d[0], v0.d[1]\n"
	"mov v3.d[0], v4.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v0.2s, v4.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v4.2d, v3.2d\n" 
	"add v0.2d, v0.2d, v1.2d\n" 
	// (v00,v01) = (v04,v05) + 2 * LOWER(v00,v01) * LOWER (v04,v05)

	"eor v12.16b, v12.16b, v0.16b\n"
	// (v12,v13) ^= (v00,v1)
	"shl v1.2d, v12.2d, #48\n"
	"ushr v3.2d, v12.2d, #16\n"
	"orr v12.16b, v3.16b, v1.16b\n"
	//  (v12,v13) = (v12,v13)>>16 | (v12,v13)<<48;

	"mov v1.d[0], v8.d[1]\n"
	"mov v3.d[0], v12.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v8.2s, v12.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v12.2d, v3.2d\n"
	"add v8.2d, v8.2d, v1.2d\n" 
	// (v08,v09) = (v12,v13) + 2 * LOWER(v08,v09) * LOWER (v12,v13)

	"eor v4.16b, v4.16b, v8.16b\n"
	// (v04.v05) ^= (v08,v09);
	"shl v1.2d, v4.2d, #1\n"
	"ushr v3.2d, v4.2d, #63\n"
	"orr v4.16b, v3.16b, v1.16b\n"
	// (v04,v05) = (v04,v05)>>63 | (v04,v05)<<1

	"mov v1.d[0], v2.d[1]\n"
	"mov v3.d[0], v6.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v2.2s, v6.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v6.2d, v3.2d\n" 
	"add v2.2d, v2.2d, v1.2d\n" 
	// (v02,v03) = (v06,v07) + 2 * LOWER(v02,v03) * LOWER (v06,v07)

	"eor v14.16b, v14.16b, v2.16b\n"
	// (v14,v15) ^= (v02,v03)
	"shl v1.2d, v14.2d, #32\n"
	"ushr v3.2d, v14.2d, #32\n"
	"orr v14.16b, v3.16b, v1.16b\n"
	// (v14,v15) = (v14,v15)>>32 | (v14,v15)<<32;

	"mov v1.d[0], v10.d[1]\n"
	"mov v3.d[0], v14.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v10.2s, v14.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v14.2d, v3.2d\n" 
	"add v10.2d, v10.2d, v1.2d\n" 
	// (v10,v11) = (v14,v15) + 2 * LOWER(v10,v11) * LOWER (v14,v15)

	"eor v6.16b, v6.16b, v10.16b\n"
	// (v06.v07) ^= (v10,v11);
	"shl v1.2d, v6.2d, #40\n"
	"ushr v3.2d, v6.2d, #24\n"
	"orr v6.16b, v3.16b, v1.16b\n"
	// (v06,v07) = (v06,v07)>>24 | (v06,v07)<<40

	"mov v1.d[0], v2.d[1]\n"
	"mov v3.d[0], v6.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v2.2s, v6.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v6.2d, v3.2d\n" 
	"add v2.2d, v2.2d, v1.2d\n" 
	// (v02,v03) = (v06,v07) + 2 * LOWER(v02,v03) * LOWER (v06,v07)

	"eor v14.16b, v14.16b, v2.16b\n"
	// (v14,v15) ^= (v02,v03)

	"shl v1.2d, v14.2d, #48\n"
	"ushr v3.2d, v14.2d, #16\n"
	"orr v14.16b, v3.16b, v1.16b\n"
	// v(v14,v15) = (v14,v15)>>16 | (v14,v15)<<48

	"mov v1.d[0], v10.d[1]\n"
	"mov v3.d[0], v14.d[1]\n"  
	"umull v1.2d, v1.2s, v3.2s\n"
	"umull v3.2d, v10.2s, v14.2s\n" 
	"mov v3.d[1], v1.d[0]\n" 
	"add v3.2d, v3.2d, v3.2d\n" 
	"add v1.2d, v14.2d, v3.2d\n" 
	"add v10.2d, v10.2d, v1.2d\n" 
	// (v10,v11) = (v14,v15) + 2 * LOWER(v10,v11) * LOWER (v14,v15)

	"eor v6.16b, v6.16b, v10.16b\n"
	// (v06,v07) ^= (v10,v11)

	"shl v1.2d, v6.2d, #1\n"
	"ushr v3.2d, v6.2d, #63\n"
	"orr v6.16b, v3.16b, v1.16b\n"
	// (v06,v07) = (v06,v07)>>63 | (v06,v07)<<1
	
	// 2nd half
	//
	// x16,x17 are scratch
	
	"mov x0, v0.d[0]\n"
	"mov x1, v0.d[1]\n"
	"mov x2, v2.d[0]\n"
	"mov x3, v2.d[1]\n"
	"mov x4, v4.d[0]\n"
	"mov x5, v4.d[1]\n"
	"mov x6, v6.d[0]\n"
	"mov x7, v6.d[1]\n"
	"mov x8, v8.d[0]\n"
	"mov x9, v8.d[1]\n"
	"mov x10, v10.d[0]\n"
	"mov x11, v10.d[1]\n"
	"mov x12, v12.d[0]\n"
	"mov x13, v12.d[1]\n"
	"mov x14, v14.d[0]\n"
	"mov x15, v14.d[1]\n"


	"umull x16, w0, w5\n"
	"add x16, x16, x16\n"
	"add x16, x5, x16\n"
	"add x0, x0, x16\n"
    	//v00 += v05 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v00))*static_cast<uint64_t>(static_cast<uint32_t>(v05));

	"eor x15, x15, x0\n" 
	"lsl x16, x15, #32\n"
	"lsr x17, x15, #32\n"
	"orr x15, x16, x17\n"
	//v15 ^= v00;
	//v15 = v15>>32 | v15<<32;

	"umull x16, w10, w15\n"	
	"add x16, x16, x16\n"
	"add x16, x15, x16\n"
	"add x10, x10, x16\n"
	//v10 += v15 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v10))*static_cast<uint64_t>(static_cast<uint32_t>(v15));
	"eor x5, x5, x10\n"
	//v05 ^= v10;
	"lsl x16, x5, #40\n"
	"lsr x17, x5, #24\n"
	"orr x5, x16, x17\n"
	//v05 = v05>>24 | v05<<40;

	"umull x16, w0, w5\n"
	"add x16, x16, x16\n"
	"add x16, x5, x16\n"
	"add x0, x0, x16\n"
	//v00 += v05 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v00))*static_cast<uint64_t>(static_cast<uint32_t>(v05));
	"eor x15, x15, x0\n" 
	"lsl x16, x15, #48\n"
	"lsr x17, x15, #16\n"
	"orr x15, x16, x17\n"
	//v15 ^= v00;
	//v15 = v15>>16 | v15<<48;

	"umull x16, w10, w15\n"	
	"add x16, x16, x16\n"
	"add x16, x15, x16\n"
	"add x10, x10, x16\n"
	//v10 += v15 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v10))*static_cast<uint64_t>(static_cast<uint32_t>(v15));
	"eor x5, x5, x10\n"
	//v05 ^= v10;
	"lsl x16, x5, #1\n"
	"lsr x17, x5, #63\n"
	"orr x5, x16, x17\n"

	"umull x16, w1, w6\n"	
	"add x16, x16, x16\n"
	"add x16, x6, x16\n"
	"add x1, x1, x16\n"
	//v01 += v06 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v01))*static_cast<uint64_t>(static_cast<uint32_t>(v06));
	"eor x12, x12, x1\n"
	//v12 ^= v01;
	"lsl x16, x12, #32\n"
	"lsr x17, x12, #32\n"
	"orr x12, x16, x17\n"
	//v12 = v12>>32 | v12<<32;
	
	"umull x16, w11, w12\n"	
	"add x16, x16, x16\n"
	"add x16, x12, x16\n"
	"add x11, x11, x16\n"
	//v11 += v12 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v11))*static_cast<uint64_t>(static_cast<uint32_t>(v12));
	"eor x6, x6, x11\n"
	//v06 ^= v11;
	"lsl x16, x6, #40\n"
	"lsr x17, x6, #24\n"
	"orr x6, x16, x17\n"
	//v06 = v06>>24 | v06<<40;
	
	"umull x16, w1, w6\n"	
	"add x16, x16, x16\n"
	"add x16, x6, x16\n"
	"add x1, x1, x16\n"
	//v01 += v06 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v01))*static_cast<uint64_t>(static_cast<uint32_t>(v06));
	"eor x12, x12, x1\n"
	//v12 ^= v01;
	"lsl x16, x12, #48\n"
	"lsr x17, x12, #16\n"
	"orr x12, x16, x17\n"
	//v12 = v12>>16 | v12<<48;

	"umull x16, w11, w12\n"	
	"add x16, x16, x16\n"
	"add x16, x12, x16\n"
	"add x11, x11, x16\n"
	//v11 += v12 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v11))*static_cast<uint64_t>(static_cast<uint32_t>(v12));
	"eor x6, x6, x11\n"
	//v06 ^= v11;
	"lsl x16, x6, #1\n"
	"lsr x17, x6, #63\n"
	"orr x6, x16, x17\n"
	//v06 = v06>>63 | v06<<1;
	
	"umull x16, w2, w7\n"	
	"add x16, x16, x16\n"
	"add x16, x7, x16\n"
	"add x2, x2, x16\n"
	//v02 += v07 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v02))*static_cast<uint64_t>(static_cast<uint32_t>(v07));
	"eor x13, x2, x13\n"
	//v13 ^= v02;
	"lsl x16, x13, #32\n"
	"lsr x17, x13, #32\n"
	"orr x13, x16, x17\n"
	//v13 = v13>>32 | v13<<32;

	"umull x16, w8, w13\n"	
	"add x16, x16, x16\n"
	"add x16, x13, x16\n"
	"add x8, x8, x16\n"
	//v08 += v13 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v08))*static_cast<uint64_t>(static_cast<uint32_t>(v13));
	"eor x7, x7, x8\n"
	//v07 ^= v08;
	"lsl x16, x7, #40\n"
	"lsr x17, x7, #24\n"
	"orr x7, x16, x17\n"
	//v07 = v07>>24 | v07<<40;

	"umull x16, w2, w7\n"	
	"add x16, x16, x16\n"
	"add x16, x7, x16\n"
	"add x2, x2, x16\n"
	//v02 += v07 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v02))*static_cast<uint64_t>(static_cast<uint32_t>(v07));
	"eor x13, x2, x13\n"
	//v13 ^= v02;
	"lsl x16, x13, #48\n"
	"lsr x17, x13, #16\n"
	"orr x13, x16, x17\n"
	//v13 = v13>>16 | v13<<48;

	"umull x16, w8, w13\n"	
	"add x16, x16, x16\n"
	"add x16, x13, x16\n"
	"add x8, x8, x16\n"
	//v08 += v13 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v08))*static_cast<uint64_t>(static_cast<uint32_t>(v13));
	"eor x7, x7, x8\n"
	//v07 ^= v08;
	"lsl x16, x7, #1\n"
	"lsr x17, x7, #63\n"
	"orr x7, x16, x17\n"
	//v07 = v07>>63 | v07<<1;

	"umull x16, w3, w4\n"	
	"add x16, x16, x16\n"
	"add x16, x4, x16\n"
	"add x3, x3, x16\n"
	//v03 += v04 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v03))*static_cast<uint64_t>(static_cast<uint32_t>(v04));
	"eor x14, x14, x3\n"
	//v14 ^= v03;
	"lsl x16, x14, #32\n"
	"lsr x17, x14, #32\n"
	"orr x14, x16, x17\n"
	//v14 = v14>>32 | v14<<32;

	"umull x16, w9, w14\n"	
	"add x16, x16, x16\n"
	"add x16, x14, x16\n"
	"add x9, x9, x16\n"
	//v09 += v14 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v09))*static_cast<uint64_t>(static_cast<uint32_t>(v14));
	"eor x4, x4, x9\n"
	//v04 ^= v09;
	"lsl x16, x4, #40\n"
	"lsr x17, x4, #24\n"
	"orr x4, x16, x17\n"
	//v04 = v04>>24 | v04<<40;

	"umull x16, w3, w4\n"	
	"add x16, x16, x16\n"
	"add x16, x4, x16\n"
	"add x3, x3, x16\n"
	//v03 += v04 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v03))*static_cast<uint64_t>(static_cast<uint32_t>(v04));
	"eor x14, x14, x3\n"
	//v14 ^= v03;
	"lsl x16, x14, #48\n"
	"lsr x17, x14, #16\n"
	"orr x14, x16, x17\n"
	//v14 = v14>>16 | v14<<48;

	"umull x16, w9, w14\n"	
	"add x16, x16, x16\n"
	"add x16, x14, x16\n"
	"add x9, x9, x16\n"
	//v09 += v14 + 2*static_cast<uint64_t>(static_cast<uint32_t>(v09))*static_cast<uint64_t>(static_cast<uint32_t>(v14));
	"eor x4, x4, x9\n"
	//v04 ^= v09;
	"lsl x16, x4, #1\n"
	"lsr x17, x4, #63\n"
	"orr x4, x16, x17\n"
	//v04 = v04>>63 | v04<<1;

	"mov v0.d[0], x0\n"
	"mov v0.d[1], x1\n"
	"mov v2.d[0], x2\n"
	"mov v2.d[1], x3\n"
	"mov v4.d[0], x4\n"
	"mov v4.d[1], x5\n"
	"mov v6.d[0], x6\n"
	"mov v6.d[1], x7\n"
	"mov v8.d[0], x8\n"
	"mov v8.d[1], x9\n"
	"mov v10.d[0], x10\n"
	"mov v10.d[1], x11\n"
	"mov v12.d[0], x12\n"
	"mov v12.d[1], x13\n"
	"mov v14.d[0], x14\n"
	"mov v14.d[1], x15\n"
