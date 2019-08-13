#include "cpu_features/include/cpu_features_cache_info.h"

#if defined(CPU_FEATURES_ARCH_X86_64)

#include "Intrinsics/X86/BlakeIntrinsics.h"

#else

#include "Intrinsics/CrossPlatform/BlakeIntrinsics.h"

#endif
