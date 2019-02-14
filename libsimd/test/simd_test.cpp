/*******************************************************
 * Copyright (C) 2016, Alexander Rusakov - All Rights Reserved
 *******************************************************/

#include <iostream>
#include "libsimd/src/simdcpuid.h"
#include "libsimd/src/simdavx.h"
#include "libsimd/src/simdsse.h"
#include "hayai/src/hayai.hpp"


#define BOOST_TEST_MODULE simdtest
#include <boost/test/unit_test.hpp>
#include "libtestutils/src/tutils.h"

using namespace std;
using namespace asr;

template <typename T> void simple_check() {
	size_t sz = 3;
	alignas(simdsse::allignment_req())  T *v1 = new T[sz];
	alignas(simdsse::allignment_req())  T *v2 = new T[sz];
    T *v3 = new T[sz];
    T *v4 = new T[sz];
    T *v3g = new T[sz];
    T *v4g = new T[sz];

	double sum(0);
	for (size_t i(0); i < sz; i++) {
		v1[i] = T(i * 0.01);
		v2[i] = T((sz - i) * 1e8);
        sum += v1[i] * v2[i];
		v3g[i] = v1[i] + v2[i];
        v4g[i] = v1[i] - v2[i];
	}
	double res = simdsse::dot_vec(v1, v2, sz);
	CHECK_CLOSE(res, sum);
	simdsse::sum(v1, v2, v3, sz);
    CHECK_CLOSE(v3, v3g, sz);
    simdsse::sub(v1, v2, v4, sz);
    CHECK_CLOSE(v4, v4g, sz);	

    delete[] v1;
    delete[] v2;
    delete[] v3;
    delete[] v4;
    delete[] v3g;
    delete[] v4g;
}


/*
BOOST_AUTO_TEST_CASE(dot_sum_sub_vec_float) {
    simple_check<float>();
}
*/


BOOST_AUTO_TEST_CASE(dot_sum_sub_vec_double) {
    simple_check<double>();
}

BOOST_AUTO_TEST_CASE(sparse_vec)
{
    size_t sz = 7;
    float *dens = new float[sz];
    double *sp_data = new double[sz];
    unsigned int *sp_idx = new unsigned[sz];

    double sum(0);
    for (size_t i(0); i < sz; i++)
    {
        dens[i] = float(i * 0.01);
        sp_data[i] = float((sz - i) * 1e8);
        sp_idx[i] = i;
        sum += dens[i] * sp_data[i];
    }

    double res = simdsse::sparse_vec_dense_vector_dot(dens, sp_data, sp_idx, sz);
    CHECK_CLOSE(res, sum);
}


BOOST_AUTO_TEST_CASE(cpuid) {
auto& outstream = std::cout;

    auto support_message = [&outstream](std::string isa_feature, bool is_supported) {
        outstream << isa_feature << (is_supported ? " supported" : " not supported") << std::endl;
    };

    InstructionSet checkIS;
    std::cout << checkIS.Vendor() << std::endl;
    std::cout << checkIS.Brand() << std::endl;

    support_message("3DNOW",       checkIS._3DNOW());
    support_message("3DNOWEXT",    checkIS._3DNOWEXT());
    support_message("ABM",         checkIS.ABM());
    support_message("ADX",         checkIS.ADX());
    support_message("AES",         checkIS.AES());
    support_message("AVX",         checkIS.AVX());
    support_message("AVX2",        checkIS.AVX2());
    support_message("AVX512CD",    checkIS.AVX512CD());
    support_message("AVX512ER",    checkIS.AVX512ER());
    support_message("AVX512F",     checkIS.AVX512F());
    support_message("AVX512PF",    checkIS.AVX512PF());
    support_message("BMI1",        checkIS.BMI1());
    support_message("BMI2",        checkIS.BMI2());
    support_message("CLFSH",       checkIS.CLFSH());
    support_message("CMPXCHG16B",  checkIS.CMPXCHG16B());
    support_message("CX8",         checkIS.CX8());
    support_message("ERMS",        checkIS.ERMS());
    support_message("F16C",        checkIS.F16C());
    support_message("FMA",         checkIS.FMA());
    support_message("FSGSBASE",    checkIS.FSGSBASE());
    support_message("FXSR",        checkIS.FXSR());
    support_message("HLE",         checkIS.HLE());
    support_message("INVPCID",     checkIS.INVPCID());
    support_message("LAHF",        checkIS.LAHF());
    support_message("LZCNT",       checkIS.LZCNT());
    support_message("MMX",         checkIS.MMX());
    support_message("MMXEXT",      checkIS.MMXEXT());
    support_message("MONITOR",     checkIS.MONITOR());
    support_message("MOVBE",       checkIS.MOVBE());
    support_message("MSR",         checkIS.MSR());
    support_message("OSXSAVE",     checkIS.OSXSAVE());
    support_message("PCLMULQDQ",   checkIS.PCLMULQDQ());
    support_message("POPCNT",      checkIS.POPCNT());
    support_message("PREFETCHWT1", checkIS.PREFETCHWT1());
    support_message("RDRAND",      checkIS.RDRAND());
    support_message("RDSEED",      checkIS.RDSEED());
    support_message("RDTSCP",      checkIS.RDTSCP());
    support_message("RTM",         checkIS.RTM());
    support_message("SEP",         checkIS.SEP());
    support_message("SHA",         checkIS.SHA());
    support_message("SSE",         checkIS.SSE());
    support_message("SSE2",        checkIS.SSE2());
    support_message("SSE3",        checkIS.SSE3());
    support_message("SSE4.1",      checkIS.SSE41());
    support_message("SSE4.2",      checkIS.SSE42());
    support_message("SSE4a",       checkIS.SSE4a());
    support_message("SSSE3",       checkIS.SSSE3());
    support_message("SYSCALL",     checkIS.SYSCALL());
    support_message("TBM",         checkIS.TBM());
    support_message("XOP",         checkIS.XOP());
    support_message("XSAVE",       checkIS.XSAVE());
}

