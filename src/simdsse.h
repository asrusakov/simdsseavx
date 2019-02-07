/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#pragma once
#include  <stddef.h>

namespace asr {
	//wrapper above sse instruction.
	//in short: sse vec is 128bit wide. It contains 4 integer, 4 floats, 2 doubles
	class simdsse {
	public:

		//returns true if sse supported on the device
		static bool is_sse_supported();
		static constexpr int allignment_req() { return SIMD_SSE_ALLIGNMENT; }

		static const int SSE_WIDTH = 128;
		static const int SSE_FLOAT_PACKED  = 4;
		static const int SSE_INT_PACKED    = 4;
		static const int SSE_DOUBLE_PACKED = 2;

		//harware requirement
		static const int SIMD_SSE_ALLIGNMENT = 16; 
		//assume 
		//d2[0:3] = d1[0:3] * d2[0:3]
		static void mul_quarks(const float *d1, float *d2);

		//v2 = v1 * v2;
		//d1 and d2 assumed to be alligned, 
		//sz >> 1, and can be any
		//function multiplies quarks  via sse and rest by the normal cycle
		static void mul_vec(const float *v1, float *v2, const size_t sz);
		//scalar product of aligned vectors of size sz
		static double dot_vec(const float *v1, const float *v2, const size_t sz);

		// dot = dense_vec[0:sz-1] * spvec_data[spvec_idxs[0:sz-1]];
		//can handle unalligned data
		static double sparse_vec_dense_vector_dot(const float  *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);
		static double sparse_vec_dense_vector_dot(const double *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);

		static bool self_test();

	protected:

	};
};
