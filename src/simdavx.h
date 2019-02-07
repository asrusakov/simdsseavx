/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#pragma once
#include  <stddef.h>

namespace asr {
	//wrapper above sse instruction.
	//in short: avx vec is 256 bit wide. It contains 8 integer, 8 floats, 4 doubles
	class simdavx {
	public:

		//returns true if sse supported on the device
		static bool is_supported();
		static constexpr int allignment_req() { return SIMD_AVX_ALLIGNMENT; }

		static const int AVX_WIDTH = 256;
		static const int AVX_FLOAT_PACKED  = 8;
		static const int AVX_INT_PACKED    = 8;
		static const int AVX_DOUBLE_PACKED = 4;

		//harware requirement
		static const int SIMD_AVX_ALLIGNMENT = 16; 

		static double sparse_vec_dense_vector_dot(const double *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);

//		static bool self_test();

	protected:
	};
};
