/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#pragma once
#include  <stddef.h>
#include "simdbase.h"

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
		//v2 = v1 * v2;
		//d1 and d2 assumed to be alligned, 
		//function multiplies quarks  via sse and rest by the normal cycle
		static void mul_vec(const float *v1, float *v2, const size_t sz);
		static void mul_vec(const double *v1, double *v2, const size_t sz);
		//scalar product of aligned vectors of size sz
		static double dot_vec(const float *v1, const float *v2, const size_t sz);


        void sum(const double *RESTRICT v1, const double *RESTRICT v2, double *RESTRICT v,  const size_t sz) ;
		void sub(const double *RESTRICT v1, const double *RESTRICT v2, double *RESTRICT v,  const size_t sz) ;

		// dot = dense_vec[0:sz-1] * spvec_data[spvec_idxs[0:sz-1]];
		//can handle unalligned data
		static double sparse_vec_dense_vector_dot(const float  *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);
		static double sparse_vec_dense_vector_dot(const double *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);

	protected:	
		//assume 
		//d2[0:3] = d1[0:3] * d2[0:3]
		static void mul_quarks(const float *d1, float *d2);

	};
}

#ifdef __SSE2__
//sse2 instructions
//#include <immintrin.h>
//#include <emmintrin.h>
//sse3
#include <pmmintrin.h>
namespace asr {
	//can handle unalligned data
	inline double simdsse::sparse_vec_dense_vector_dot(const double *RESTRICT  dense_vec,
														const double  *RESTRICT spvec_data, 
														const unsigned int *RESTRICT  spvec_idxs, size_t sz) {
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

/*
does not work now
	//can handle unalligned data
	double simdsse::sparse_vec_dense_vector_dot(const float *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz) {
		const int fourPacks = SSE_FLOAT_PACKED;
		const size_t tsz = sz - sz % fourPacks;
		__m128d acc2d = _mm_setzero_pd();
		alignas(simdsse::allignment_req()) float tmp[fourPacks];
		size_t idx[SSE_FLOAT_PACKED];
		for (size_t i(0); i < tsz; i += fourPacks) {
			//load to tmp
			idx[0] = *pidx++;
			idx[1] = *pidx++;
			idx[2] = *pidx++;
			idx[3] = *pidx++;
			__m128d spvec = _mm_set_ps (spvec_data[idx[1]], spvec_data[idx[0]], spvec_data[idx[1]], spvec_data[idx[0]]);			

			const __m128 dvec  = _mm_loadu_ps(&dense_vec[i]); 
			spvec = _mm_mul_ps(spvec, dvec); 
			acc2d = _mm_add_pd(acc2d, halfsum_m128(spvec));
		}
		double sum = sum_m128d(acc2d);
		for (size_t i(tsz >= 0 ? tsz : 0); i < sz; i++) {
			sum += spvec_data[spvec_idxs[i]] * dense_vec[i];
		}
		return sum;
	}
*/

	inline void simdsse::mul_vec(const float *v1, float *v2, const size_t sz) {
		const size_t tsz = sz - sz % SSE_FLOAT_PACKED;
		for (size_t i(0); i < tsz; i += SSE_FLOAT_PACKED) {
	 		__m128 s1 = _mm_loadu_ps(v1);
			__m128 s2 = _mm_loadu_ps(v2);
			s2 = _mm_mul_ps(s1, s2);
			_mm_storeu_ps(v2, s2);
			v1 += SSE_FLOAT_PACKED;
			v2 += SSE_FLOAT_PACKED;
		}
		for (size_t i(tsz >= 0 ? tsz : 0); i < sz; i++) v2[i] *= v1[i];
	}

	inline void simdsse::mul_vec(const double *v1, double *v2, const size_t sz) {
		const size_t tsz = sz - sz % SSE_DOUBLE_PACKED;
		for (size_t i(0); i < tsz; i += SSE_DOUBLE_PACKED) {
	 		__m128d s1 = _mm_loadu_pd(v1);
			__m128d s2 = _mm_loadu_pd(v2);
			s2 = _mm_mul_pd(s1, s2);
			_mm_storeu_pd(v2, s2);
			v1 += SSE_DOUBLE_PACKED;
			v2 += SSE_DOUBLE_PACKED;
		}
		for (size_t i(tsz >= 0 ? tsz : 0); i < sz; i++) v2[i] *= v1[i];
	}


	//v = v1 + v2
	//runtime: native sum: 22s, avx sum with storage 14, sum with stream to memory 29
	inline void simdsse::sum(const double *RESTRICT v1, const double *RESTRICT v2, double *RESTRICT v,  const size_t sz) {
		const size_t tsz = sz - sz % SSE_DOUBLE_PACKED;
		size_t i(0);
	
		for (; i < tsz; i += SSE_DOUBLE_PACKED) {
			//strange but not alligned access to unalligned data does not crash and works fine. why?
			const __m128d v1vec = _mm_loadu_pd(v1);
			const __m128d v2vec = _mm_loadu_pd(v2);
			v1 += SSE_DOUBLE_PACKED;
			v2 += SSE_DOUBLE_PACKED;
			__m128d s = _mm_add_pd(v1vec, v2vec);
			_mm_storeu_pd(v, s);

			v += SSE_DOUBLE_PACKED;
		}		
		for ( ; i < sz; i++) *v++ = *v1++ * *v2++;
	}

	//v = v1 - v2
	//runtime: native sum: 22s, avx sum with storage 14, sum with stream to memory 19
	inline void simdsse::sub(const double *RESTRICT v1, const double *RESTRICT v2, double *RESTRICT v, const size_t sz) {
		const size_t tsz = sz - sz % SSE_DOUBLE_PACKED;
		size_t i(0);

		for (; i < tsz; i += SSE_DOUBLE_PACKED) {
			//strange but not alligned access to unalligned data does not crash and works fine. why?
			const __m128d v1vec = _mm_loadu_pd(v1);
			const __m128d v2vec = _mm_loadu_pd(v2);
			v1 += SSE_DOUBLE_PACKED;
			v2 += SSE_DOUBLE_PACKED;
			__m128d s = _mm_sub_pd(v1vec, v2vec);
			_mm_storeu_pd(v, s);
			v += SSE_DOUBLE_PACKED;
		}
		for (; i < sz; i++) *v++ = *v1++ * *v2++;
	}

};
#endif
