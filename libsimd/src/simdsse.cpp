/***************************
**
** utils for running iterative matrix solution over SSE.
**
*/
#ifdef __SSE2__

#include "simdsse.h"

//implementatin
//sse2 instructions
#include<immintrin.h>


#include <assert.h>
#include <math.h>

namespace asr {
	long int simdsse::simdentry = 0;
	long int simdsse::simdentry_parallel = 0;
	long double simdsse::sum_row_sz = 0.0;

	bool simdsse::is_supported() {
		//supposed to look into cpuid
		return true;
	}

    inline 	double sum_m128d(const __m128d &v) {
		double tmp[2];
		_mm_store_pd(tmp, v);
		return tmp[0] + tmp[1];
	}

	inline __m128d halfsum_m128(const __m128 &v) {
		const __m128 t   = _mm_add_ps(v, _mm_movehl_ps(v, v));
		//const __m128 ss = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
		const __m128d r = _mm_cvtps_pd(t);
		return r;
	}

	double sum_m128(const __m128 &v) {
		//const __m128 t   = _mm_add_ps(v, _mm_movehl_ps(v, v));
		//const __m128 ss = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
		float tmp[4];
		_mm_store_ps(tmp, v);
		double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
		return sum;
	}

	void simdsse::mul_quarks(const float *v1, float *v2) {
		__m128 s1 = _mm_load_ps(v1);
		__m128 s2 = _mm_load_ps(v2);
		__m128 s3 = _mm_mul_ps(s1, s2);
		_mm_store_ps(v2, s3);
	}

	double dot_quarks(const float *v1, const float *v2) {
		__m128 s1 = _mm_load_ps(v1);
		__m128 s2 = _mm_load_ps(v2);
		__m128 s3 = _mm_mul_ps(s1, s2);
		return sum_m128(s3); //SLOW
	}

	//dot = v1[0:16] * v2[0:16] 
	__m128 dot_4quarks(const float *v1, const float *v2) {
		__m128 s11 = _mm_load_ps(v1);
		__m128 s12 = _mm_load_ps(v2);
		__m128 d1 = _mm_mul_ps(s11, s12);

		__m128 s21 = _mm_load_ps(v1 + 4);
		__m128 s22 = _mm_load_ps(v2 + 4);
		__m128 d2 = _mm_mul_ps(s21, s22);

		__m128 s31 = _mm_load_ps(v1 + 8);
		__m128 s32 = _mm_load_ps(v2 + 8);
		__m128 d3 = _mm_mul_ps(s31, s32);

		__m128 s41 = _mm_load_ps(v1 + 12);
		__m128 s42 = _mm_load_ps(v2 + 12);
		__m128 d4 = _mm_mul_ps(s41, s42);

		d1 = _mm_add_ps(d1, d2);
		d3 = _mm_add_ps(d3, d4);
		d1 = _mm_add_ps(d1, d3);

		return d1;
	}

	void simdsse::mul_vec(const float *v1, float *v2, const size_t sz) {
		const size_t tsz = sz - sz % SSE_FLOAT_PACKED;
		for (size_t i(0); i < tsz; i += SSE_FLOAT_PACKED) mul_quarks(v1 + i, v2 + i);
		for (size_t i(tsz >= 0 ? tsz : 0); i < sz; i++) v2[i] *= v1[i];
	}

	//can handle unalligned data
	double simdsse::sparse_vec_dense_vector_dot(const float *dense_vec, const double *spvec_data, const unsigned int *spvec_idxs, size_t sz) {
		const int fourPacks = SSE_FLOAT_PACKED;
		const size_t tsz = sz - sz % fourPacks;
		__m128d acc2d = _mm_setzero_pd();
		alignas(simdsse::allignment_req()) float tmp[fourPacks];
		for (size_t i(0); i < tsz; i += fourPacks) {
			//load to tmp
			tmp[0] = float(spvec_data[spvec_idxs[i]]);
			tmp[1] = float(spvec_data[spvec_idxs[i+1]]);
			tmp[2] = float(spvec_data[spvec_idxs[i+2]]);
			tmp[3] = float(spvec_data[spvec_idxs[i+3]]);
			__m128 spvec = _mm_load_ps(tmp);
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


	//assumed allignment
	double simdsse::dot_vec(const float *v1, const float *v2, const size_t sz) {
		const int fourPacks = SSE_FLOAT_PACKED * 4;
		const size_t tsz = sz - sz % fourPacks;
		__m128d acc2d = _mm_setzero_pd();
		for (size_t i(0); i < tsz; i += fourPacks) {
			__m128 sum4 = dot_4quarks(v1 + i, v2 + i);
			//we may continue summing of quarks but i am afraid we will loose ortogonality
			acc2d = _mm_add_pd(acc2d, halfsum_m128(sum4));
		}
		double sum = sum_m128d(acc2d);
		for (size_t i(tsz >= 0 ? tsz : 0); i < sz; i++) sum += v2[i] * v1[i];
		return sum;
	}


	template <typename T> void CHECK_CLOSE(T a, T b) {
		const double eps = 1e-6;
		if (fabs(a - b) > fabs(eps*(a+b)) + eps) 
			throw "error in check close";
	}

	template <typename T> void CHECK_CLOSE(T *a, T *b, size_t sz) {
		for (size_t i(0); i < sz; i++) CHECK_CLOSE(a[i], b[i]);
	}

	//some trivial test for simple subroutines
	bool simdsse::self_test() {
		const double eps = 1e-8;
		alignas(simdsse::allignment_req())  float ex[4] = { 1.0, 100.0, 1000.0, 10000.0 };
		double exsum = ex[0] + ex[1] + ex[2] + ex[3];
		{
			//test summing of a quark
			__m128 q1 = _mm_load_ps(ex);
			double val = sum_m128(q1);
			CHECK_CLOSE(val, exsum);

			const auto t = halfsum_m128(q1);
			double r2 = sum_m128d(t);
			CHECK_CLOSE(r2, exsum);
		}

		{
			assert(is_supported());
		}

		{
			alignas(simdsse::allignment_req())  float d2[4] = { -1.0, -100.0, -1000.0, -10000.0 };;
			float tmp[] = { d2[0] * ex[0], d2[1]*ex[1], d2[2] * ex[2], d2[3] * ex[3] };
			mul_quarks(ex, d2);
			CHECK_CLOSE(d2, tmp, 4);
		}

		{
			size_t sz = 173;
			alignas(simdsse::allignment_req())  float *v1 = new float[sz];
			alignas(simdsse::allignment_req())  float *v2 = new float[sz];
			double sum(0);
			for (size_t i(0); i < sz; i++) {
				v1[i] = float(i * 0.01);
				v2[i] = float((sz - i) * 1e8);
				sum += v1[i] * v2[i];
			}

			double res = simdsse::dot_vec(v1, v2, sz);
			CHECK_CLOSE(res, sum);
		}

		{
			size_t sz = 7;
			float  *dens= new float[sz];
			double *sp_data = new double[sz];
			unsigned int    *sp_idx  = new unsigned [sz];

			double sum(0);
			for (size_t i(0); i < sz; i++) {
				dens[i] = float(i * 0.01);
				sp_data[i] = float((sz - i) * 1e8);
				sp_idx[i] = i;
				sum += dens[i] * sp_data[i];
			}

			double res = simdsse::sparse_vec_dense_vector_dot(dens, sp_data, sp_idx, sz);
			CHECK_CLOSE(res, sum);
		}


		return true;
	}

}

#endif