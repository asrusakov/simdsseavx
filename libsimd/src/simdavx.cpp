/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#include "simdavx.h"
//implementatin
//sse2 instructions
#include "simdcpuid.h"

namespace asr {
	bool simdavx::is_supported() {
#ifndef __AVX__ 
		return false;
#else 	
		InstructionSet isa;
#ifdef __AVX2__ 		
		return isa.AVX2();			
#else  		
		return isa.AVX();			
#endif	
#endif	
	}

}
