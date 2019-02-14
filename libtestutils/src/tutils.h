/***************************
**
** some trivial utils used in the quicktests
**
*/
#pragma once

template <typename T> void CHECK_CLOSE(T a, T b, const double eps = 1e-6) {
	if (fabs(a - b) > fabs(eps*(a+b)) + eps) 
		throw "error in check close";
}


template <typename T> void CHECK_CLOSE(T *a, T *b, size_t sz, const double eps = 1e-6) {
	for (size_t i(0); i < sz; i++) CHECK_CLOSE(a[i], b[i], eps);
}

//Prevent compiler of optimizing out code 
//ref CppCon 2015: Chandler Carruth "Tuning C++: Benchmarks, and CPUs, and Compilers! Oh My!"
void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

void clobber() {
  asm volatile("" : : : "memory");
}
