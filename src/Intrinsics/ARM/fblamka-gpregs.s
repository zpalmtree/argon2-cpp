"umull x16, w0, w4\n"
"add x16, x4, x16, lsl #1\n"
"add x0, x0, x16\n"
"umull x16, w1, w5\n"
"add x16, x5, x16, lsl #1\n"
"add x1, x1, x16\n"
"umull x16, w2, w6\n"
"add x16, x6, x16, lsl #1\n"
"add x2, x2, x16\n"
"umull x16, w3, w7\n"
"add x16, x7, x16, lsl #1\n"
"add x3, x3, x16\n"

"eor x12, x12, x0\n"
"lsr x16, x12, #32\n"
"lsl x17, x12, #32\n"
"orr x12, x16, x17\n"

"eor x13, x13, x1\n"
"lsr x16, x13, #32\n"
"lsl x17, x13, #32\n"
"orr x13, x16, x17\n"


"umull x16, w8, w12\n"
"add x16, x12, x16, lsl #1\n"
"add x8, x8, x16\n"
"eor x4, x4, x8\n"
"lsr x16, x4, #24\n"
"lsl x17, x4, #40\n"
"orr x4, x16, x17\n"


"umull x16, w0, w4\n"
"add x16, x4, x16, lsl #1\n"
"add x0, x0, x16\n"
"eor x12, x12, x0\n"
"lsr x16, x12, #16\n"
"lsl x17, x12, #48\n"
"orr x12, x16, x17\n"


"umull x16, w8, w12\n"
"add x16, x12, x16, lsl #1\n"
"add x8, x8, x16\n"
"eor x4, x4, x8\n"
"lsr x16, x4, #63\n"
"lsl x17, x4, #1\n"
"orr x4, x16, x17\n"


"umull x16, w9, w13\n"
"add x16, x13, x16, lsl #1\n"
"add x9, x9, x16\n"
"eor x5, x5, x9\n"
"lsr x16, x5, #24\n"
"lsl x17, x5, #40\n"
"orr x5, x16, x17\n"


"umull x16, w1, w5\n"
"add x16, x5, x16, lsl #1\n"
"add x1, x1, x16\n"
"eor x13, x13, x1\n"
"lsr x16, x13, #16\n"
"lsl x17, x13, #48\n"
"orr x13, x16, x17\n"

"umull x16, w9, w13\n"
"add x16, x13, x16, lsl #1\n"
"add x9, x9, x16\n"

"eor x5, x5, x9\n"
"lsr x16, x5, #63\n"
"lsl x17, x5, #1\n"
"orr x5, x16, x17\n"

"eor x14, x14, x2\n"
"lsr x16, x14, #32\n"
"lsl x17, x14, #32\n"
"orr x14, x16, x17\n"

"eor x15, x15, x3\n"
"lsr x16, x15, #32\n"
"lsl x17, x15, #32\n"
"orr x15, x16, x17\n"


"umull x16, w10, w14\n"
"add x16, x14, x16, lsl #1\n"
"add x10, x10, x16\n"
"umull x16, w11, w15\n"
"add x16, x15, x16, lsl #1\n"
"add x11, x11, x16\n"

"eor x6, x6, x10\n"
"lsr x16, x6, #24\n"
"lsl x17, x6, #40\n"
"orr x6, x16, x17\n"


"umull x16, w2, w6\n"
"add x16, x6, x16, lsl #1\n"
"add x2, x2, x16\n"
"eor x14, x14, x2\n"
"lsr x16, x14, #16\n"
"lsl x17, x14, #48\n"
"orr x14, x16, x17\n"


"umull x16, w10, w14\n"
"add x16, x14, x16, lsl #1\n"
"add x10, x10, x16\n"

"eor x6, x6, x10\n"
"lsr x16, x6, #63\n"
"lsl x17, x6, #1\n"
"orr x6, x16, x17\n"

"eor x7, x7, x11\n"
"lsr x16, x7, #24\n"
"lsl x17, x7, #40\n"
"orr x7, x16, x17\n"


"umull x16, w3, w7\n"
"add x16, x7, x16, lsl #1\n"
"add x3, x3, x16\n"
"eor x15, x15, x3\n"
"lsr x16, x15, #16\n"
"lsl x17, x15, #48\n"
"orr x15, x16, x17\n"


"umull x16, w11, w15\n"
"add x16, x15, x16, lsl #1\n"
"add x11, x11, x16\n"
"eor x7, x7, x11\n"
"lsr x16, x7, #63\n"
"lsl x17, x7, #1\n"
"orr x7, x16, x17\n"


"umull x16, w0, w5\n"
"add x16, x5, x16, lsl #1\n"
"add x0, x0, x16\n"
"umull x16, w1, w6\n"
"add x16, x6, x16, lsl #1\n"
"add x1, x1, x16\n"
"umull x16, w2, w7\n"
"add x16, x7, x16, lsl #1\n"
"add x2, x2, x16\n"
"umull x16, w3, w4\n"
"add x16, x4, x16, lsl #1\n"
"add x3, x3, x16\n"

"eor x15, x15, x0\n"
"lsr x16, x15, #32\n"
"lsl x17, x15, #32\n"
"orr x15, x16, x17\n"


"umull x16, w10, w15\n"
"add x16, x15, x16, lsl #1\n"
"add x10, x10, x16\n"
"eor x5, x5, x10\n"
"lsr x16, x5, #24\n"
"lsl x17, x5, #40\n"
"orr x5, x16, x17\n"


"umull x16, w0, w5\n"
"add x16, x5, x16, lsl #1\n"
"add x0, x0, x16\n"
"eor x15, x15, x0\n"
"lsr x16, x15, #16\n"
"lsl x17, x15, #48\n"
"orr x15, x16, x17\n"

"umull x16, w10, w15\n"
"add x16, x15, x16, lsl #1\n"
"add x10, x10, x16\n"

"eor x5, x5, x10\n"
"lsr x16, x5, #63\n"
"lsl x17, x5, #1\n"
"orr x5, x16, x17\n"

"eor x12, x12, x1\n"
"lsr x16, x12, #32\n"
"lsl x17, x12, #32\n"
"orr x12, x16, x17\n"


"umull x16, w11, w12\n"
"add x16, x12, x16, lsl #1\n"
"add x11, x11, x16\n"
"eor x6, x6, x11\n"
"lsr x16, x6, #24\n"
"lsl x17, x6, #40\n"
"orr x6, x16, x17\n"


"umull x16, w1, w6\n"
"add x16, x6, x16, lsl #1\n"
"add x1, x1, x16\n"
"eor x12, x12, x1\n"
"lsr x16, x12, #16\n"
"lsl x17, x12, #48\n"
"orr x12, x16, x17\n"

"umull x16, w11, w12\n"
"add x16, x12, x16, lsl #1\n"
"add x11, x11, x16\n"
"eor x6, x6, x11\n"
"lsr x16, x6, #63\n"
"lsl x17, x6, #1\n"
"orr x6, x16, x17\n"


"eor x13, x13, x2\n"
"lsr x16, x13, #32\n"
"lsl x17, x13, #32\n"
"orr x13, x16, x17\n"

"umull x16, w8, w13\n"
"add x16, x13, x16, lsl #1\n"
"add x8, x8, x16\n"
"eor x7, x7, x8\n"
"lsr x16, x7, #24\n"
"lsl x17, x7, #40\n"
"orr x7, x16, x17\n"


"umull x16, w2, w7\n"
"add x16, x7, x16, lsl #1\n"
"add x2, x2, x16\n"
"eor x13, x13, x2\n"
"lsr x16, x13, #16\n"
"lsl x17, x13, #48\n"
"orr x13, x16, x17\n"

"umull x16, w8, w13\n"
"add x16, x13, x16, lsl #1\n"
"add x8, x8, x16\n"
"eor x7, x7, x8\n"
"lsr x16, x7, #63\n"
"lsl x17, x7, #1\n"
"orr x7, x16, x17\n"


"eor x14, x14, x3\n"
"lsr x16, x14, #32\n"
"lsl x17, x14, #32\n"
"orr x14, x16, x17\n"

"umull x16, w9, w14\n"
"add x16, x14, x16, lsl #1\n"
"add x9, x9, x16\n"
"eor x4, x4, x9\n"
"lsr x16, x4, #24\n"
"lsl x17, x4, #40\n"
"orr x4, x16, x17\n"


"umull x16, w3, w4\n"
"add x16, x4, x16, lsl #1\n"
"add x3, x3, x16\n"
"eor x14, x14, x3\n"
"lsr x16, x14, #16\n"
"lsl x17, x14, #48\n"
"orr x14, x16, x17\n"

"umull x16, w9, w14\n"
"add x16, x14, x16, lsl #1\n"
"add x9, x9, x16\n"
"eor x4, x4, x9\n"
"lsr x16, x4, #63\n"
"lsl x17, x4, #1\n"
"orr x4, x16, x17\n"

