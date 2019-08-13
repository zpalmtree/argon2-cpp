#include "cpu_features/include/cpu_features_cache_info.h"

#if defined(CPU_FEATURES_ARCH_X86_64)

#warning "Using X86_64 Blake AVX2 Intrinsics"

#include "Intrinsics/X86/BlakeIntrinsics.h"

#else

#warning "Using cross platform Blake code"

#include "Intrinsics/CrossPlatform/BlakeIntrinsics.h"

#endif
