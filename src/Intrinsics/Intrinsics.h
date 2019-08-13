#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_WIN64))

#include "Intrinsics/X86/BlakeIntrinsics.h"

#else

#include "Intrinsics/CrossPlatform/BlakeIntrinsics.h"

#endif
