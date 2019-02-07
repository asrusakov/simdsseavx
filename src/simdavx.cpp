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

	//can handle unalligned data
	double simdavx::sparse_vec_dense_vector_dot(const double *__restrict__ dense_vec, const double *__restrict__ spvec_data, const unsigned int *__restrict__ spvec_idxs, size_t sz) {
		const int fourPacks = AVX_DOUBLE_PACKED;
		const size_t tsz = sz - sz % AVX_DOUBLE_PACKED;
		__m256d acc4d = _mm256_setzero_pd();
		alignas(simdavx::allignment_req()) double tmp[fourPacks];
		for (size_t i(0); i < tsz; i += fourPacks) {
			//load to tmp
			tmp[0] = spvec_data[spvec_idxs[i]];
			tmp[1] = spvec_data[spvec_idxs[i+1]];
			tmp[2] = spvec_data[spvec_idxs[i+2]];
			tmp[3] = spvec_data[spvec_idxs[i+3]];

			__m256d spvec = _mm256_load_pd(tmp);            
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
		for (size_t i(tsz >= 0 ? tsz : 0); i < sz; i++) {
			sum += spvec_data[spvec_idxs[i]] * dense_vec[i];
		}
		return sum;
	}
}