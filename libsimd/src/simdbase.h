/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#pragma once
#include  <stddef.h>


#ifdef WIN32
#define RESTRICT
#else
//#define RESTRICT 
#define RESTRICT __restrict__
#endif

namespace asr {
	//wrapper above sse instruction.
	//TISA - is eiter simdsse or simdavx
	template <class TISA> class simd {
	public:

		//returns true if sse supported on the device
		static bool          is_supported()   { return TISA::is_supported(); };
		static constexpr int allignment_req() { return TISA::allignment_req();}
        
		static double dot_vec(const double *v1, const double *v2, const size_t sz) { return TISA::dot_vec(v1, v2, sz); };        
		static void   sum(const double * v1, const double * v2, double * v, const size_t sz) {return TISA:: sum(v1, v2, v, sz);}
		static void   sub(const double *v1, const double *v2, double * v, const size_t sz)   {return TISA:: sub(v1, v2, v, sz);} 

		// dot = dense_vec[0:sz-1] * spvec_data[spvec_idxs[0:sz-1]];
		//can handle unalligned data
		static double sparse_vec_dense_vector_dot(const float  *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz) {
            return TISA::sparse_vec_dense_vector_dot(dense_vec, spvec_data, spvec_idxs, sz);
        };
		static double sparse_vec_dense_vector_dot(const double *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz) {
            return TISA::sparse_vec_dense_vector_dot(dense_vec, spvec_data, spvec_idxs, sz);
        };

	protected:	
	};
}
