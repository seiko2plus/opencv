// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/hal/intrin.hpp"

// ARITHM_DISPATCHING_ONLY defined by arithm dispatch file

#undef ARITHM_DECLARATIONS_ONLY
#ifdef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
    #define ARITHM_DECLARATIONS_ONLY
#endif

#undef ARITHM_DEFINITIONS_ONLY
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && !defined(ARITHM_DISPATCHING_ONLY)
    #define ARITHM_DEFINITIONS_ONLY
#endif

namespace cv { namespace hal {

#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#endif

//=======================================
// Arithmetic and logical operations
// +, -, *, /, &, |, ^, ~, abs ...
//=======================================

#ifdef ARITHM_DEFINITIONS_ONLY

////////////////////////////////////////////////////////////////////////////////

// Add
template<typename T1, typename Tvec>
struct op_add
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a + b; }
    static T1 r(T1 a, T1 b)
    { return a + b; }
};
template<typename T1, typename Tvec>
struct op_adds
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a + b; }
    static T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(a + b); }
};
// Subtract
template<typename T1, typename Tvec>
struct op_sub
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a - b; }
    static T1 r(T1 a, T1 b)
    { return a - b; }
};
template<typename T1, typename Tvec>
struct op_subs
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a - b; }
    static T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(a - b); }
};
// Max & Min
template<typename T1, typename Tvec>
struct op_max
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_max(a, b); }
    static T1 r(T1 a, T1 b)
    { return std::max(a, b); }
};
template<typename T1, typename Tvec>
struct op_min
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_min(a, b); }
    static T1 r(T1 a, T1 b)
    { return std::min(a, b); }
};
// Absolute difference
template<typename T1, typename Tvec>
struct op_absd
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static T1 r(T1 a, T1 b)
    { return std::abs(a - b); }
};
// 'specializations to prevent "-0" results'
template<typename Tvec>
struct op_absd<float, Tvec>
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static float r(float a, float b)
    { return std::abs(a - b); }
};
template<typename Tvec>
struct op_absd<double, Tvec>
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static double r(double a, double b)
    { return std::abs(a - b); }
};
template<typename T1, typename Tvec>
struct op_absd_int
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_reinterpret_as_s32(v_absdiff(a, b)); }
    static T1 r(T1 a, T1 b)
    { return a > b ? a - b : b - a; }
};
template<typename T1, typename Tvec>
struct op_absds_u
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(std::abs(a - b)); }
};
template<typename T1, typename Tvec>
struct op_absds_s
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiffs(a, b); }
    static T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(std::abs(a - b)); }
};
// Logical
template<typename T1, typename Tvec>
struct op_or
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a | b; }
    static T1 r(T1 a, T1 b)
    { return a | b; }
};
template<typename T1, typename Tvec>
struct op_xor
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a ^ b; }
    static T1 r(T1 a, T1 b)
    { return a ^ b; }
};
template<typename T1, typename Tvec>
struct op_and
{
    static Tvec r(const Tvec& a, const Tvec& b)
    { return a & b; }
    static T1 r(T1 a, T1 b)
    { return a & b; }
};

template< template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct op_loader
{
    typedef V_RegTraits<Tvec> trait;
    typedef typename trait::v128_reg Tvec128;
    typedef OP<T1, Tvec> op;
    typedef OP<T1, Tvec128> op128;

    static void run(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec a = vx_load(src1);
        Tvec b = vx_load(src2);
        v_store(dst, op::r(a, b));
    }
    static void run_align(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec a = vx_load_aligned(src1);
        Tvec b = vx_load_aligned(src2);
        v_store_aligned(dst, op::r(a, b));
    }
    static void run_f128(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec128 a = v_load(src1);
        Tvec128 b = v_load(src2);
        v_store(dst, op128::r(a, b));
    }
    static void run_f64(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec128 a = v_load_low(src1);
        Tvec128 b = v_load_low(src2);
        v_store_low(dst, op128::r(a, b));
    }
    static T1 run(T1 a, T1 b)
    { return op::r(a, b); }
};

// void src2
template<typename T1, typename Tvec>
struct op_not
{
    typedef V_RegTraits<Tvec> trait;
    typedef typename trait::v128_reg Tvec128;

    static void run(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load(src1);
        v_store(dst, ~a);
    }
    static void run_align(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load_aligned(src1);
        v_store_aligned(dst, ~a);
    }
    static void run_f128(const T1* src1, const T1*, T1* dst)
    {
        Tvec128 a = v_load(src1);
        v_store(dst, ~a);
    }
    static void run_f64(const T1* src1, const T1*, T1* dst)
    {
        Tvec128 a = v_load_low(src1);
        v_store_low(dst, ~a);
    }
    static T1 run(T1 a, T1)
    { return ~a; }
};

#endif // ARITHM_DEFINITIONS_ONLY

//////////////////////////// Generate loops /////////////////////////////////

#define SIMD_C_FUN simd_bin
#define SIMD_C_RF128 1
#define SIMD_C_RF64 1
#define SIMD_C_SIMD128_UNROLL !CV_NEON
#include "arithm_helper.hpp"

#define SIMD_C_FUN simd_bin32
#define SIMD_C_RALIGN !CV_NEON
#define SIMD_C_RF128 1
#define SIMD_C_RF64 1
#define SIMD_C_SIMD128_UNROLL !CV_NEON
#include "arithm_helper.hpp"

#define SIMD_C_FUN simd_bin64
#define SIMD_C_SIMD CV_SIMD_64F
#define SIMD_C_RALIGN !CV_NEON
#define SIMD_C_RF128 1
#define SIMD_C_SIMD128_UNROLL !CV_NEON
#define SIMD_C_CPP_UNROLL 1
#include "arithm_helper.hpp"

////////////////////////////////////////////////////////////////////////////////////

#undef SIMD_DECLARE_OP
#define SIMD_DECLARE_OP(fun, tp) void fun(SIMD_ARGS(tp, tp));

#undef SIMD_DEFINE_BIN
#define SIMD_DEFINE_BIN(simd_fun, fun_name, c_type, v_type, op_name) \
    void fun_name(SIMD_ARGS(c_type, c_type))                         \
    {                                                                \
        CV_INSTRUMENT_REGION();                                      \
        simd_fun<op_loader<op_name, c_type, v_type>, c_type, c_type> \
        (SIMD_ARGS_PASS);                                            \
    }

#undef SIMD_DISPATCH_OP
#define SIMD_DISPATCH_OP(fun_name, c_type, v_type, op_name)                     \
    void fun_name(SIMD_ARGS(c_type, c_type), void*)                             \
    {                                                                           \
        CV_INSTRUMENT_REGION();                                                 \
        CALL_HAL(fun_name, __CV_CAT(cv_hal_, fun_name), SIMD_ARGS_PASS)         \
        ARITHM_CALL_IPP(__CV_CAT(arithm_ipp_, fun_name), SIMD_ARGS_PASS)        \
        CV_CPU_DISPATCH(fun_name, (SIMD_ARGS_PASS), CV_CPU_DISPATCH_MODES_ALL); \
    }

// 8bit - 16bit
#undef SIMD_DEFINE_OP
#define SIMD_DEFINE_OP(fun_name, c_type, v_type, op_name) \
    SIMD_DEFINE_BIN(simd_bin, fun_name, c_type, v_type, op_name)

SIMD_DEF_SAT(add, op_adds)
SIMD_DEF_SAT(sub, op_subs)
SIMD_DEF_SAT(min, op_min)
SIMD_DEF_SAT(max, op_max)

SIMD_DEF_U8(absdiff, op_absds_u)
SIMD_DEF_S8(absdiff, op_absds_s)
SIMD_DEF_U16(absdiff, op_absds_u)
SIMD_DEF_S16(absdiff, op_absds_s)

SIMD_DEF_U8(or,  op_or)
SIMD_DEF_U8(xor, op_xor)
SIMD_DEF_U8(and, op_and)

// 32-bit
#undef SIMD_DEFINE_OP
#define SIMD_DEFINE_OP(fun_name, c_type, v_type, op_name) \
    SIMD_DEFINE_BIN(simd_bin32, fun_name, c_type, v_type, op_name)

SIMD_DEF_NSAT(add, op_add)
SIMD_DEF_NSAT(sub, op_sub)
SIMD_DEF_NSAT(min, op_min)
SIMD_DEF_NSAT(max, op_max)
SIMD_DEF_S32(absdiff, op_absd_int)
SIMD_DEF_F32(absdiff, op_absd)

// 64-bit
#undef SIMD_DEFINE_OP
#define SIMD_DEFINE_OP(fun_name, c_type, v_type, op_name) \
    SIMD_DEFINE_BIN(simd_bin64, fun_name, c_type, v_type, op_name)

SIMD_DEF_F64(add, op_add)
SIMD_DEF_F64(sub, op_sub)
SIMD_DEF_F64(min, op_min)
SIMD_DEF_F64(max, op_max)
SIMD_DEF_F64(absdiff, op_absd)

////////////////////////////////////////////////////////////////////////////////

// One source!, an exception for operation "not"
// we could use macros here but it's better to implement it
// with that way to give more clarification
// about how macroS "SIMD_DEF_*" are works

#if defined(ARITHM_DECLARATIONS_ONLY) || defined(ARITHM_DEFINITIONS_ONLY)
void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
#endif

#ifdef ARITHM_DEFINITIONS_ONLY
void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    CV_INSTRUMENT_REGION();
    simd_bin<op_not<uchar, v_uint8>, uchar, uchar>(src1, step1, src2, step2, dst, step, width, height);
}
#endif

#ifdef ARITHM_DISPATCHING_ONLY
void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height, void*)
{
    CV_INSTRUMENT_REGION();
    CALL_HAL(not8u, cv_hal_not8u, src1, step1, dst, step, width, height)
    ARITHM_CALL_IPP(arithm_ipp_not8u, src1, step1, dst, step, width, height)
    CV_CPU_DISPATCH(not8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}
#endif

#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_END
#endif

}} // cv::hal::