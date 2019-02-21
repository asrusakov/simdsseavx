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
	//in short: avx vec is 256 bit wide. It contains 8 integer, 8 floats, 4 doubles
	class simdavx {
	public:

		//returns true if sse supported on the device
		static bool is_supported();
		static constexpr int allignment_req() { return SIMD_AVX_ALLIGNMENT; }

		//harware requirement
		static const int SIMD_AVX_ALLIGNMENT = 32; 

		static double sparse_vec_dense_vector_dot(const double *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);
		static double sparse_vec_dense_vector_dot(const float  *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz);

		static double dot_vec(const double *v1, const double *v2, const size_t sz);
		static void   sum(const double * v1, const double * v2, double * v, const size_t sz);
		static void   sub(const double *v1, const double *v2, double * v, const size_t sz);

	protected:
		static const int AVX_WIDTH = 256;
		static const int AVX_FLOAT_PACKED  = 8;
		static const int AVX_INT_PACKED    = 8;
		static const int AVX_DOUBLE_PACKED = 4;
	};
};


#ifdef __AVX__
//avx instructions
#include <immintrin.h>

namespace asr {

	//can handle unalligned data
	inline double simdavx::sparse_vec_dense_vector_dot(const double * RESTRICT dense_vec,
												const double *RESTRICT spvec_data,
												const unsigned int *RESTRICT spvec_idxs, size_t sz) {
		const int fourPacks = AVX_DOUBLE_PACKED;
		const size_t tsz = sz - sz % AVX_DOUBLE_PACKED;
		__m256d acc4d = _mm256_setzero_pd();
		auto pidx = spvec_idxs;

        unsigned int idx[4];
//		alignas(simdavx::allignment_req()) double tmp[fourPacks];
		size_t i(0);
		for (; i < tsz; i += fourPacks) {      
            idx[0]=*pidx++;
            idx[1]=*pidx++;
            idx[2]=*pidx++;
            idx[3]=*pidx++;
			__m256d spvec = _mm256_set_pd(spvec_data[idx[3]],
										  spvec_data[idx[2]],
										  spvec_data[idx[1]],
										  spvec_data[idx[0]]);

			const __m256d dvec  = _mm256_loadu_pd(&dense_vec[i]);     
#ifdef __AVX2__
			//for avx2 should be:
			// FMA: rowsum += x_ * v_
			acc4d = _mm256_fmadd_pd(spvec, dvec, acc4d);
#else
			spvec = _mm256_mul_pd(spvec, dvec); 
			acc4d = _mm256_add_pd(acc4d, spvec);
#endif
		}        
        acc4d = _mm256_hadd_pd(acc4d, acc4d);
        double sum = ((double*)&acc4d)[0] + ((double*)&acc4d)[2];
		_mm256_zeroupper();
		for ( ; i < sz; i++) {
			sum += spvec_data[*pidx++] * dense_vec[i];
		}
		return sum;
	}	

	//can handle unalligned data
	inline double simdavx::sparse_vec_dense_vector_dot(const float  * RESTRICT dense_vec,
												const double *RESTRICT spvec_data,
												const unsigned int *RESTRICT spvec_idxs, size_t sz) {
		const size_t tsz = (tsz > AVX_DOUBLE_PACKED) ?
						sz - sz % AVX_DOUBLE_PACKED - AVX_DOUBLE_PACKED : 0;
		__m256d acc4d = _mm256_setzero_pd();
		auto pidx = spvec_idxs;

        unsigned int idx[4];
		size_t i(0);
		for (; i < tsz; i += AVX_DOUBLE_PACKED) {      
            idx[0]=*pidx++;
            idx[1]=*pidx++;
            idx[2]=*pidx++;
            idx[3]=*pidx++;
			__m256d spvec = _mm256_set_pd(spvec_data[idx[3]],
										  spvec_data[idx[2]],
										  spvec_data[idx[1]],
										  spvec_data[idx[0]]);

			//HERE MIGHT be a penalty for sse/avx transition
			const __m128  dvecf  = _mm_loadu_ps(&dense_vec[i]);     
			const __m256d dvec   = _mm256_cvtps_pd(dvecf); 			
#ifdef __AVX2__
			//for avx2 should be:
			// FMA: rowsum += x_ * v_
			acc4d = _mm256_fmadd_pd(spvec, dvec, acc4d);
#else
			spvec = _mm256_mul_pd(spvec, dvec); 
			acc4d = _mm256_add_pd(acc4d, spvec);
#endif
		}        
        acc4d = _mm256_hadd_pd(acc4d, acc4d);
        double sum = ((double*)&acc4d)[0] + ((double*)&acc4d)[2];
		_mm256_zeroupper();
		for ( ; i < sz; i++) {
			sum += spvec_data[*pidx++] * dense_vec[i];
		}
		return sum;
	}


	inline double simdavx::dot_vec(const double *RESTRICT v1, const double *RESTRICT v2, const size_t sz) {
		const size_t tsz = sz - sz % AVX_DOUBLE_PACKED;
		__m256d acc4d = _mm256_setzero_pd();
		size_t i(0);
		for (; i < tsz; i += AVX_DOUBLE_PACKED) {
			//strange but not alligned access to unalligned data does not crash and works fine. why?
			__m256d v1vec = _mm256_loadu_pd(v1);
			const __m256d v2vec = _mm256_loadu_pd(v2);

			//			const __m256d v1vec = _mm256_loadu_pd(v1);
			//			const __m256d v2vec = _mm256_loadu_pd(v2);

			v1 += AVX_DOUBLE_PACKED;
			v2 += AVX_DOUBLE_PACKED;
#ifdef __AVX2__
			//for avx2 should be:
			// FMA: rowsum += x_ * v_
			acc4d = _mm256_fmadd_pd(v1vec, v2vec, acc4d);
#else
			v1vec = _mm256_mul_pd(v1vec, v2vec);
			acc4d = _mm256_add_pd(acc4d, v1vec);
#endif
		}
		acc4d = _mm256_hadd_pd(acc4d, acc4d);
		double sum = ((double*)&acc4d)[0] + ((double*)&acc4d)[2];
		_mm256_zeroupper();
		for (; i < sz; i++) sum += *v1++ * *v2++;
		return sum;
	}

	//v = v1 + v2
	//runtime: native sum: 22s, avx sum with storage 14, sum with stream to memory 29
	inline void simdavx::sum(const double *RESTRICT v1, const double *RESTRICT v2, double *RESTRICT v,  const size_t sz) {
		const size_t tsz = sz - sz % AVX_DOUBLE_PACKED;
		size_t i(0);
	
		for (; i < tsz; i += AVX_DOUBLE_PACKED) {
			//strange but not alligned access to unalligned data does not crash and works fine. why?
			const __m256d v1vec = _mm256_loadu_pd(v1);
			const __m256d v2vec = _mm256_loadu_pd(v2);
			v1 += AVX_DOUBLE_PACKED;
			v2 += AVX_DOUBLE_PACKED;
			__m256d s = _mm256_add_pd(v1vec, v2vec);
			_mm256_storeu_pd(v, s);
//			_mm256_stream_pd(v, s);
			v += AVX_DOUBLE_PACKED;
		}
		_mm256_zeroupper();
		for ( ; i < sz; i++) *v++ = *v1++ * *v2++;
	}

	//v = v1 + v2
	//runtime: native sum: 22s, avx sum with storage 14, sum with stream to memory 19
	inline void simdavx::sub(const double *RESTRICT v1, const double *RESTRICT v2, double *RESTRICT v, const size_t sz) {
		const size_t tsz = sz - sz % AVX_DOUBLE_PACKED;
		size_t i(0);

		for (; i < tsz; i += AVX_DOUBLE_PACKED) {
			//strange but not alligned access to unalligned data does not crash and works fine. why?
			const __m256d v1vec = _mm256_loadu_pd(v1);
			const __m256d v2vec = _mm256_loadu_pd(v2);
			v1 += AVX_DOUBLE_PACKED;
			v2 += AVX_DOUBLE_PACKED;
			__m256d s = _mm256_sub_pd(v1vec, v2vec);
			_mm256_storeu_pd(v, s);
			//			_mm256_stream_pd(v, s);
			v += AVX_DOUBLE_PACKED;
		}
		_mm256_zeroupper();
		for (; i < sz; i++) *v++ = *v1++ * *v2++;
	}
}
#endif