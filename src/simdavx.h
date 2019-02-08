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


//avx instructions
#include <immintrin.h>

namespace asr {

	//can handle unalligned data
	double simdavx::sparse_vec_dense_vector_dot(const double * dense_vec, 
												const double * spvec_data,
												const unsigned int *spvec_idxs, size_t sz) {
		const int fourPacks = AVX_DOUBLE_PACKED;
		const size_t tsz = sz - sz % AVX_DOUBLE_PACKED;
		__m256d acc4d = _mm256_setzero_pd();
		auto pidx = spvec_idxs;

//		alignas(simdavx::allignment_req()) double tmp[fourPacks];
		size_t i(0);
		for (; i < tsz; i += fourPacks) {
			//load to tmp. How do it better? in avx2 there is a gather but we are in avx1
/*			tmp[0] = spvec_data[spvec_idxs[i]];
			tmp[1] = spvec_data[spvec_idxs[i+1]];
			tmp[2] = spvec_data[spvec_idxs[i+2]];
			tmp[3] = spvec_data[spvec_idxs[i+3]];
			__m256d spvec = _mm256_load_pd(tmp);            
*/
			__m256d spvec = _mm256_set_pd(spvec_data[*(pidx + 3)],
										  spvec_data[*(pidx + 2)],
										  spvec_data[*(pidx + 1)],
										  spvec_data[*(pidx + 0)]);
			pidx+=4;

			const __m256d dvec  = _mm256_loadu_pd(&dense_vec[i]);             
			spvec = _mm256_mul_pd(spvec, dvec); 
			acc4d = _mm256_add_pd(acc4d, spvec);

            //for avx2 should be:
            // FMA: rowsum += x_ * v_
            //acc4d = _mm256_fmadd_pd(spvec_, dvec_, acc4d);
		}        
        acc4d = _mm256_hadd_pd(acc4d, acc4d);
        double sum = ((double*)&acc4d)[0] + ((double*)&acc4d)[2];

        //avoid avx/sse transition penalties
        _mm256_zeroupper();
		for ( ; i < sz; i++) {
			sum += spvec_data[spvec_idxs[i]] * dense_vec[i];
		}
		return sum;
	}
}