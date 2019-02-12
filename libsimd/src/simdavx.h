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

#ifdef __AVX__
//avx instructions
#include <immintrin.h>

namespace asr {

	//can handle unalligned data
	inline double simdavx::sparse_vec_dense_vector_dot(const double * __restrict__ dense_vec, 
												const double *__restrict__ spvec_data,
												const unsigned int *__restrict__ spvec_idxs, size_t sz) {
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

			const __m256d dvec  = _mm256_load_pd(&dense_vec[i]);             
			spvec = _mm256_mul_pd(spvec, dvec); 
			acc4d = _mm256_add_pd(acc4d, spvec);

            //for avx2 should be:
            // FMA: rowsum += x_ * v_
            //acc4d = _mm256_fmadd_pd(spvec_, dvec_, acc4d);
		}        
        acc4d = _mm256_hadd_pd(acc4d, acc4d);
        double sum = ((double*)&acc4d)[0] + ((double*)&acc4d)[2];

		for ( ; i < sz; i++) {
			sum += spvec_data[spvec_idxs[i]] * dense_vec[i];
		}
		return sum;
	}
}

#endif