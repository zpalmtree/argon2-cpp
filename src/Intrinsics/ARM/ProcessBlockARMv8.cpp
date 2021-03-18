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
    void processBlockARMv8NoXor(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock)
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

	"noxor_initloop:\n"
	"ldp x0, x1, [x21], #16\n"
	"ldp x2, x3, [x21], #16\n"
	"ldp x4, x5, [x21], #16\n"
	"ldp x6, x7, [x21], #16\n"
	"ldp x8, x9, [x21], #16\n"
	"ldp x10, x11, [x21], #16\n"
	"ldp x12, x13, [x21], #16\n"
	"ldp x14, x15, [x21], #16\n" //x0..16 is initialized with ref

	"ldp x16,x17, [x22], #16\n" //pref
	"eor x0, x0, x16\n"
	"eor x1, x1, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x2, x2, x16\n"
	"eor x3, x3, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x4, x4, x16\n"
	"eor x5, x5, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x6, x6, x16\n"
	"eor x7, x7, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x8, x8, x16\n"
	"eor x9, x9, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x10, x10, x16\n"
	"eor x11, x11, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x12, x12, x16\n"
	"eor x13, x13, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x14, x14, x16\n"
	"eor x15, x15, x17\n"
	// state is all lined up in x0...x15

#include "fblamka-gpregs.s"

    "stp x0, x1, [x20], #16\n"
    "stp x2, x3, [x20], #16\n"
    "stp x4, x5, [x20], #16\n"
    "stp x6, x7, [x20], #16\n"
    "stp x8, x9, [x20], #16\n"
    "stp x10, x11, [x20], #16\n"
    "stp x12, x13, [x20], #16\n"
    "stp x14, x15, [x20], #16\n"

	"subs w24, w24, #1\n"
	"bne noxor_initloop\n"

	"mov x24, #8\n"
	"subs x20,x20, #1024\n"
	"noxor_mainloop:\n"

	"ldp x0, x1, [x20], #128\n"
	"ldp x2, x3, [x20], #128\n"
	"ldp x4, x5, [x20], #128\n"
	"ldp x6, x7, [x20], #128\n"
	"ldp x8, x9, [x20], #128\n"
	"ldp x10, x11, [x20], #128\n"
	"ldp x12, x13, [x20], #128\n"
	"ldp x14, x15, [x20], #128\n" //x0..x15 is initialized with ref
	"subs x20, x20, #1024\n"

#include "fblamka-gpregs.s"

    "stp x0, x1, [x20], #128\n"
    "stp x2, x3, [x20], #128\n"
    "stp x4, x5, [x20], #128\n"
    "stp x6, x7, [x20], #128\n"
    "stp x8, x9, [x20], #128\n"
    "stp x10, x11, [x20], #128\n"
    "stp x12, x13, [x20], #128\n"
    "stp x14, x15, [x20], #128\n"

    	"subs x20, x20, #1008\n"
	"subs w24, w24, #1\n"
	"bne noxor_mainloop\n"

	"mov w4, #16\n"
	"ldr x0, %[next]\n"
	"subs x21, x21, #1024\n"
	"subs x22, x22, #1024\n"
	"ldr x20, %[state]\n"

	"noxor_loop:\n"
	"ld4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x22], #64\n" // prevBlock
        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x20], #64\n" // state

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x21], #64\n" // refBlock

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

	"st4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0], #64\n"

	"subs w4, w4, #1\n"
	"bne noxor_loop\n"

	: : [state] "m" (state), [prev] "m" (prev), [ref] "m" (ref), [next] "m" (next)
	: "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
	"cc", "x20", "x21", "x22", "x24");
}
    void processBlockARMv8DoXor(
        Block &nextBlock,
        const Block &refBlock,
        const Block &prevBlock)
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

	"doxor_initloop:\n"
	"ldp x0, x1, [x21], #16\n"
	"ldp x2, x3, [x21], #16\n"
	"ldp x4, x5, [x21], #16\n"
	"ldp x6, x7, [x21], #16\n"
	"ldp x8, x9, [x21], #16\n"
	"ldp x10, x11, [x21], #16\n"
	"ldp x12, x13, [x21], #16\n"
	"ldp x14, x15, [x21], #16\n" //x0..16 is initialized with ref

	"ldp x16,x17, [x22], #16\n" //pref
	"eor x0, x0, x16\n"
	"eor x1, x1, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x2, x2, x16\n"
	"eor x3, x3, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x4, x4, x16\n"
	"eor x5, x5, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x6, x6, x16\n"
	"eor x7, x7, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x8, x8, x16\n"
	"eor x9, x9, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x10, x10, x16\n"
	"eor x11, x11, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x12, x12, x16\n"
	"eor x13, x13, x17\n"
	"ldp x16,x17, [x22], #16\n" //pref
	"eor x14, x14, x16\n"
	"eor x15, x15, x17\n"
	// state is all lined up in x0...x15

#include "fblamka-gpregs.s"

    "stp x0, x1, [x20], #16\n"
    "stp x2, x3, [x20], #16\n"
    "stp x4, x5, [x20], #16\n"
    "stp x6, x7, [x20], #16\n"
    "stp x8, x9, [x20], #16\n"
    "stp x10, x11, [x20], #16\n"
    "stp x12, x13, [x20], #16\n"
    "stp x14, x15, [x20], #16\n"

	"subs w24, w24, #1\n"
	"bne doxor_initloop\n"

	"mov x24, #8\n"
	"subs x20,x20, #1024\n"
	"doxor_mainloop:\n"

	"ldp x0, x1, [x20], #128\n"
	"ldp x2, x3, [x20], #128\n"
	"ldp x4, x5, [x20], #128\n"
	"ldp x6, x7, [x20], #128\n"
	"ldp x8, x9, [x20], #128\n"
	"ldp x10, x11, [x20], #128\n"
	"ldp x12, x13, [x20], #128\n"
	"ldp x14, x15, [x20], #128\n" //x0..x15 is initialized with ref
	"subs x20, x20, #1024\n"

#include "fblamka-gpregs.s"

    "stp x0, x1, [x20], #128\n"
    "stp x2, x3, [x20], #128\n"
    "stp x4, x5, [x20], #128\n"
    "stp x6, x7, [x20], #128\n"
    "stp x8, x9, [x20], #128\n"
    "stp x10, x11, [x20], #128\n"
    "stp x12, x13, [x20], #128\n"
    "stp x14, x15, [x20], #128\n"

    	"subs x20, x20, #1008\n"
	"subs w24, w24, #1\n"
	"bne doxor_mainloop\n"

	"mov w4, #16\n"
	"ldr x0, %[next]\n"
	"subs x21, x21, #1024\n"
	"subs x22, x22, #1024\n"
	"subs x20, x20, #128\n"

	"xorloop:\n"
	"ld4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x22], #64\n" // prevBlock
        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x20], #64\n" // state

	"eor v0.16b, v0.16b, v4.16b\n"
	"eor v1.16b, v1.16b, v5.16b\n"
	"eor v2.16b, v2.16b, v6.16b\n"
	"eor v3.16b, v3.16b, v7.16b\n"

        "ld4 {v4.2d, v5.2d, v6.2d, v7.2d}, [x21], #64\n" // refBlock

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
	: "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
	"cc", "x20", "x21", "x22", "x24");
}

}

