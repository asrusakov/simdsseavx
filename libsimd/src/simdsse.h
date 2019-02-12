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
		static bool is_supported();
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

		//debug stats
		static long int simdentry;
		static long int simdentry_parallel;
		static long double sum_row_sz;

	protected:
	};
}

//sse2 instructions
//#include <immintrin.h>
//#include <emmintrin.h>
//sse3
#include <pmmintrin.h>
namespace asr {
	//can handle unalligned data
	inline double simdsse::sparse_vec_dense_vector_dot(const double *__restrict__  dense_vec, const double  *__restrict__ spvec_data, const unsigned int *__restrict__  spvec_idxs, size_t sz) {
		const unsigned int fourPacks = SSE_DOUBLE_PACKED;
		const size_t tsz = sz - sz % SSE_DOUBLE_PACKED;
		__m128d acc2d = _mm_setzero_pd();
		auto pidx = spvec_idxs;
		size_t i(0);
		size_t idx[2];
		for (; i < tsz; i += fourPacks) {
			//very slow
			idx[0] = *pidx++;
			idx[1] = *pidx++;

			__m128d spvec = _mm_set_pd (spvec_data[idx[1]], spvec_data[idx[0]]);			
			const __m128d dvec  = _mm_loadu_pd(&dense_vec[i]); 
			acc2d = _mm_add_pd(acc2d, _mm_mul_pd(spvec, dvec));
		}
		acc2d =  _mm_hadd_pd(acc2d,acc2d);
		double sum = _mm_cvtsd_f64(acc2d);		
		//double sum = ((double*)&acc2d)[0] + ((double*)&acc2d)[1];
				
		for ( ; i < sz; i++) {
			sum += spvec_data[*pidx++] * dense_vec[i];
		}		
		return sum;
	}


};
