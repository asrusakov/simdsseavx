/*******************************************************
 * Copyright (C) 2016, Alexander Rusakov - All Rights Reserved
 *******************************************************/

#include <iostream>
#include "libsimd/src/simdcpuid.h"

#define BOOST_TEST_MODULE simdtest
#include <boost/test/unit_test.hpp>

using namespace std;
using namespace asr;

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

