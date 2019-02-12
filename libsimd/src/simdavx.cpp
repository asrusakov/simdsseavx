/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#include "simdavx.h"

//implementatin
//sse2 instructions
#include <immintrin.h>

namespace asr {
	bool simdavx::is_supported() {
		//supposed to look into cpuid
		return true;
	}

}
