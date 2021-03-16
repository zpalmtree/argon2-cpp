/*
 * ARMv8 optimized Argon2
*/

// Copyright (c) 2019, notasailor
//
// Please see the included LICENSE file for more information.

////////////////////////////////////////////
#include "Intrinsics/ARM/ProcessBlockARMv8.h"
////////////////////////////////////////////

#include <cstring>

namespace ProcessBlockARMv8
{
    void processBlockARMv8(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock,
        const bool doXor)
{
    static Block _state;

    uint64_t *state = _state.data();
    const uint64_t *prev = prevBlock.data();
    const uint64_t *ref = refBlock.data();
    uint64_t *next = nextBlock.data();

    asm volatile(
	"mov w24, #8\n"
	"ldr x20, %[state]\n"
	"ldr x21, %[ref]\n"
	"ldr x22, %[prev]\n"

	"initloop:\n"
	"ld1 {v4.2d, v5.2d, v6.2d, v7.2d}, [x21], #64\n" //ref
	"ld1 {v8.2d, v9.2d, v10.2d, v11.2d}, [x22], #64\n" //prev

	"eor v0.16b, v4.16b, v8.16b\n"
	"eor v2.16b, v5.16b, v9.16b\n"
	"eor v4.16b, v6.16b, v10.16b\n"
	"eor v6.16b, v7.16b, v11.16b\n" 

	"ld1 {v12.2d, v13.2d, v14.2d, v15.2d}, [x21], #64\n"
	"ld1 {v16.2d, v17.2d, v18.2d, v19.2d}, [x22], #64\n"

	"eor v8.16b, v12.16b, v16.16b\n"
	"eor v10.16b, v13.16b, v17.16b\n"
	"eor v12.16b, v14.16b, v18.16b\n"
	"eor v14.16b, v15.16b, v19.16b\n"
	// state is all lined up in v0,v2,v4...v14

#include "ARMv8BlamkaCore.s"
	"mov v16.16b, v0.16b\n"
	"mov v17.16b, v2.16b\n"
	"mov v18.16b, v4.16b\n"
	"mov v19.16b, v6.16b\n"
	"st1 {v16.2d, v17.2d, v18.2d, v19.2d}, [x20], #64\n"
	"mov v16.16b, v8.16b\n"
	"mov v17.16b, v10.16b\n"
	"mov v18.16b, v12.16b\n"
	"mov v19.16b, v14.16b\n"
	"st1 {v16.2d, v17.2d, v18.2d, v19.2d}, [x20], #64\n"

	"subs w24, w24, #1\n"
	"bne initloop\n"

	"mov x24, #0\n"
	"ldr x20, %[state]\n"
	"mainloop:\n"

	"ld1 {v0.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v2.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v4.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v6.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v8.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v10.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v12.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"ld1 {v14.2d}, [x20]\n"
	"subs x20, x20, #896\n"

#include "ARMv8BlamkaCore.s"

	"st1 {v0.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v2.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v4.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v6.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v8.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v10.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v12.2d}, [x20]\n"
	"add x20, x20, #128\n"
	"st1 {v14.2d}, [x20]\n"
	"subs x20, x20, #896\n"

	"add x24, x24, #16\n"
	"add x20, x20, #16\n"
	"cmp x24, #128\n"
	"bne mainloop\n"

	: : [state] "m" (state), [prev] "m" (prev), [ref] "m" (ref), [next] "m" (next)
       	: "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "v1", "v3", "v16", "v17", "v16", "v17", "v18", "v19",
	  "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17",
	"cc", "x20", "x21", "x22", "x24");

    if (doXor)
    {
	asm volatile(
	"mov w4, #16\n"
	"ldr x0, %[next]\n"
	"ldr x1, %[ref]\n"
	"ldr x2, %[prev]\n"
	"ldr x3, %[state]\n"

	"xorloop:\n"
	"ld4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x2], #64\n" // prevBlock
        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x3], #64\n" // state

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x1], #64\n" // refBlock

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x0]\n" // nextBlock

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

	"st4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0], #64\n"

	"subs w4, w4, #1\n"
	"bne xorloop\n"

	: : [state] "m" (state), [prev] "m" (prev), [ref] "m" (ref), [next] "m" (next)
       	: "cc", "x0", "x1", "x2", "x3", "w4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"  );

    }
    else
    {
	asm volatile(
	"mov w4, #16\n"
	"ldr x0, %[next]\n"
	"ldr x1, %[ref]\n"
	"ldr x2, %[prev]\n"
	"ldr x3, %[state]\n"

	"noxorloop:\n"
	"ld4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x2], #64\n" // prevBlock
        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x3], #64\n" // state

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x1], #64\n" // refBlock

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

	"st4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0], #64\n"

	"subs w4, w4, #1\n"
	"bne noxorloop\n"

	: : [state] "m" (state), [prev] "m" (prev), [ref] "m" (ref), [next] "m" (next)
       	: "cc", "x0", "x1", "x2", "x3", "w4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"  );
    }
}
}

