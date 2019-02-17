/*******************************************************
 * Copyright (C) 2019, Alexander Rusakov - All Rights Reserved
 *******************************************************/
#include <vector>
#include <iostream>
#include "libsimd/src/simdcpuid.h"
#include "libsimd/src/simdavx.h"
#include "libsimd/src/simdsse.h"
#include "hayai/src/hayai.hpp"
#include "libtestutils/src/tutils.h"

using namespace std;
using namespace asr;

class RUNTIME_EXP  {
public:
	RUNTIME_EXP(size_t size) {
		sz = size;
		Init();
	}

	void Init() {
		v1f.resize(sz);
		v2f.resize(sz);
		v3f.resize(sz);

		v1d.resize(sz);
		v2d.resize(sz);
		v3d.resize(sz);
		for (size_t i(0); i < sz; i++) {
			v1f[i] = float(i * 0.01);
			v2f[i] = float((sz - i) * 1e8);

			v1d[i] = double(i * 0.01);
			v2d[i] = double((sz - i) * 1e8);
		}
	}

	double dot_float_native();
	double dot_double_native();
	double dot_float();
	double dot_double();
	void   sum_float();
	void   sum_double();
	void   sum_float_native();
	void   sum_double_native();
	double sub_float();
	double sub_double();

	std::vector<float> v1f, v2f, v3f;
	std::vector<double> v1d, v2d, v3d;	
	size_t sz;
};


double RUNTIME_EXP::dot_float() {	
	return simdsse::dot_vec(&v1f[0], &v2f[0], sz);
}


double RUNTIME_EXP::dot_double() {	
	return simdsse::dot_vec(&v1d[0], &v2d[0], sz);
}



double RUNTIME_EXP::dot_float_native() {	
	double s(0);
	for(size_t i(0);i<sz;i++) s += v1f[i] * v2f[i] ;
	return s;
}


double RUNTIME_EXP::dot_double_native() {	
	double s(0);
	for(size_t i(0);i<sz;i++) s += v1d[i] * v2d[i] ;
	return s;
}

void RUNTIME_EXP::sum_double_native() {	
	for(size_t i(0);i<sz;i++) v3d[i] += v1d[i] * v2d[i] ;	
}

void RUNTIME_EXP::sum_double() {	
	return simdsse::sum(&v1d[0], &v2d[0], &v3d[0], sz);
}

void RUNTIME_EXP::sum_float_native() {	
	for(size_t i(0);i<sz;i++) v3f[i] += v1f[i] * v2f[i] ;	
}

void RUNTIME_EXP::sum_float() {	
	//simdsse::sum(&v1f[0], &v2f[0], &v3f[0], sz);
}


RUNTIME_EXP *runtime_exp = 0;
const int NITERATIONS = 2000;
const int NRUNS       = 5;

BENCHMARK(dot_native, dot_double, NRUNS, NITERATIONS) {
    runtime_exp->dot_double_native();
}

BENCHMARK(dot, dot_double, NRUNS, NITERATIONS) {
    runtime_exp->dot_double();
}

BENCHMARK(dot, dot_float_native, NRUNS, NITERATIONS) {
    runtime_exp->dot_float_native();
}

BENCHMARK(dot, dot_float, NRUNS, NITERATIONS) {
    runtime_exp->dot_float();
}


BENCHMARK(sum, sum_double_native, NRUNS, NITERATIONS) {
    runtime_exp->sum_double_native();
}

BENCHMARK(sum, sum_double, NRUNS, NITERATIONS) {
    runtime_exp->sum_double();
}

int main()
{ 	
	runtime_exp = new RUNTIME_EXP(500000);

    hayai::ConsoleOutputter consoleOutputter;
 
    hayai::Benchmarker::AddOutputter(consoleOutputter);
    hayai::Benchmarker::RunAllTests();

	delete runtime_exp;
    return 0;
}