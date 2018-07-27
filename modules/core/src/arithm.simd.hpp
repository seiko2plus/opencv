// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace hal {

CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

//=======================================
// Arithmetic and logical operations
// +, -, *, /, &, |, ^, ~, abs ...
//=======================================

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#define ARITHM_DEF_OP_BIN_R(fun, sfx, pfx, _Tvec, _op)              \
    static inline void fun(const T1* src1, const T1* src2, T1* dst) \
    {                                                               \
        _Tvec a = sfx##load##pfx(src1);                             \
        _Tvec b = sfx##load##pfx(src2);                             \
        v_store##pfx(dst, _op);                                     \
    }

#define ARITHM_IMPL_OP_BIN(name, _op, _vop)                         \
    template<typename T1, typename Tvec>                            \
    struct op_##name                                                \
    {                                                               \
        typedef V_RegTraits<Tvec> trait;                            \
        typedef typename trait::v128_reg Tvec128;                   \
        ARITHM_DEF_OP_BIN_R(run,  vx_, /*void*/, Tvec, _vop)        \
        ARITHM_DEF_OP_BIN_R(run_align, vx_, _aligned, Tvec, _vop)   \
        ARITHM_DEF_OP_BIN_R(run_f128, v_,/*void*/, Tvec128, _vop)   \
        ARITHM_DEF_OP_BIN_R(run_f64, v_, _low, Tvec128, _vop)       \
        static inline T1 run(T1 a, T1 b)                            \
        { return _op; }                                             \
    };

// symmetric operator
#define ARITHM_IMPL_OP_BIN_SYM(name, _op) ARITHM_IMPL_OP_BIN(name, _op, _op)

// Add
ARITHM_IMPL_OP_BIN_SYM(add, a + b)
ARITHM_IMPL_OP_BIN(adds, saturate_cast<T1>(a + b), a + b)
// Subtract
ARITHM_IMPL_OP_BIN_SYM(sub, (a - b))
ARITHM_IMPL_OP_BIN(subs, saturate_cast<T1>(a - b), a - b)
// Max & Min
ARITHM_IMPL_OP_BIN(max, std::max(a, b), v_max(a, b))
ARITHM_IMPL_OP_BIN(min, std::min(a, b), v_min(a, b))
// AbsDiff
ARITHM_IMPL_OP_BIN(absds_u, saturate_cast<T1>(std::abs(a - b)), v_absdiff(a, b))
ARITHM_IMPL_OP_BIN(absds_s, saturate_cast<T1>(std::abs(a - b)), v_absdiffs(a, b))
ARITHM_IMPL_OP_BIN(absd_int, (a > b ? a - b : b - a), v_reinterpret_as_s32(v_absdiff(a, b)))
// todo: investigate on 'specializations to prevent "-0" results'
ARITHM_IMPL_OP_BIN(absd_f, std::abs(a - b), v_absdiff(a, b))
// Logical
ARITHM_IMPL_OP_BIN_SYM(or,  a | b)
ARITHM_IMPL_OP_BIN_SYM(xor, a ^ b)
ARITHM_IMPL_OP_BIN_SYM(and, a & b)

// void src2
template<typename T1, typename Tvec>
struct op_not
{
    typedef V_RegTraits<Tvec> trait;
    typedef typename trait::v128_reg Tvec128;

    static inline void run(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load(src1);
        v_store(dst, ~a);
    }
    static inline void run_align(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load_aligned(src1);
        v_store_aligned(dst, ~a);
    }
    static inline void run_f128(const T1* src1, const T1*, T1* dst)
    {
        Tvec128 a = v_load(src1);
        v_store(dst, ~a);
    }
    static inline void run_f64(const T1* src1, const T1*, T1* dst)
    {
        Tvec128 a = v_load_low(src1);
        v_store_low(dst, ~a);
    }
    static inline T1 run(T1 a, T1)
    { return ~a; }
};

#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

////////////////////////////////////////////////////////////////////////////////

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
#define SIMD_C_RALIGN !CV_NEON
#define SIMD_C_CPP_UNROLL 1
#if CV_SIMD_64F
    #define SIMD_C_RF128 1
    #define SIMD_C_SIMD128_UNROLL !CV_NEON
#else
    #define SIMD_C_SIMD 0
#endif
#include "arithm_helper.hpp"

////////////////////////////////////////////////////////////////////////////////

#undef SIMD_DECLARE_OP
#define SIMD_DECLARE_OP(fun, tp) void fun(SIMD_ARGS(tp, tp));

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#define SIMD_DEFINE_BIN(simd_fun, fun_name, c_type, v_type, op_name)       \
    void fun_name(SIMD_ARGS(c_type, c_type))                               \
    {                                                                      \
        CV_INSTRUMENT_REGION();                                            \
        simd_fun<op_name<c_type, v_type>, c_type, c_type>(SIMD_ARGS_PASS); \
    }

#define SIMD_DEFINE_OP(fun_name, c_type, v_type, op_name) \
    SIMD_DEFINE_BIN(simd_bin, fun_name, c_type, v_type, op_name)
#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

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
SIMD_DEF_U8(not, op_not)

// 32-bit
#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#undef SIMD_DEFINE_OP
#define SIMD_DEFINE_OP(fun_name, c_type, v_type, op_name) \
    SIMD_DEFINE_BIN(simd_bin32, fun_name, c_type, v_type, op_name)
#endif

SIMD_DEF_NSAT(add, op_add)
SIMD_DEF_NSAT(sub, op_sub)
SIMD_DEF_NSAT(min, op_min)
SIMD_DEF_NSAT(max, op_max)
SIMD_DEF_S32(absdiff, op_absd_int)
SIMD_DEF_F32(absdiff, op_absd_f)

// 64-bit
#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#undef SIMD_DEFINE_OP
#define SIMD_DEFINE_OP(fun_name, c_type, v_type, op_name) \
    SIMD_DEFINE_BIN(simd_bin64, fun_name, c_type, v_type, op_name)
#endif

SIMD_DEF_F64(add, op_add)
SIMD_DEF_F64(sub, op_sub)
SIMD_DEF_F64(min, op_min)
SIMD_DEF_F64(max, op_max)
SIMD_DEF_F64(absdiff, op_absd_f)

////////////////////////////////////////////////////////////////////////////////


CV_CPU_OPTIMIZATION_NAMESPACE_END

}} // cv::hal::