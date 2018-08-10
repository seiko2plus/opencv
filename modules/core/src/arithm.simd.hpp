// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/core/hal/intrin.hpp"

//=========================================
// Declare & Define & Dispatch in one step
//=========================================

// ARITHM_DISPATCHING_ONLY defined by arithm dispatch file

#undef ARITHM_DECLARATIONS_ONLY
#ifdef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
    #define ARITHM_DECLARATIONS_ONLY
#endif

#undef ARITHM_DEFINITIONS_ONLY
#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY) && !defined(ARITHM_DISPATCHING_ONLY)
    #define ARITHM_DEFINITIONS_ONLY
#endif

#ifdef ARITHM_DECLARATIONS_ONLY
    #undef DEFINE_SIMD
    #define DEFINE_SIMD(fun_name, c_type, ...) \
        DECLARE_SIMD_FUN(fun_name, c_type)
#endif // ARITHM_DECLARATIONS_ONLY

#ifdef ARITHM_DEFINITIONS_ONLY
    #undef DEFINE_SIMD
    #define DEFINE_SIMD(fun_name, c_type, v_type, ...)          \
        DECLARE_SIMD_FUN(fun_name, c_type)                      \
        DEFINE_SIMD_FUN(fun_name, c_type, v_type, __VA_ARGS__)
#endif // ARITHM_DEFINITIONS_ONLY

#ifdef ARITHM_DISPATCHING_ONLY
    #undef DEFINE_SIMD
    #define DEFINE_SIMD(fun_name, c_type, v_type, ...)           \
        DISPATCH_SIMD_FUN(fun_name, c_type, v_type, __VA_ARGS__)
#endif // ARITHM_DISPATCHING_ONLY

// workaround when neon miss support of double precision
#undef DEFINE_NOSIMD
#ifdef ARITHM_DEFINITIONS_ONLY
    #define DEFINE_NOSIMD(fun_name, c_type, ...)          \
        DECLARE_SIMD_FUN(fun_name, c_type)                \
        DEFINE_NOSIMD_FUN(fun_name, c_type, __VA_ARGS__)
#else
    #define DEFINE_NOSIMD DEFINE_SIMD
#endif // ARITHM_DEFINITIONS_ONLY

#ifndef SIMD_GUARD

#define DEFINE_SIMD_U8(fun, ...) \
    DEFINE_SIMD(__CV_CAT(fun, 8u), uchar, v_uint8, __VA_ARGS__)

#define DEFINE_SIMD_S8(fun, ...) \
    DEFINE_SIMD(__CV_CAT(fun, 8s), schar, v_int8,  __VA_ARGS__)

#define DEFINE_SIMD_U16(fun, ...) \
    DEFINE_SIMD(__CV_CAT(fun, 16u), ushort, v_uint16, __VA_ARGS__)

#define DEFINE_SIMD_S16(fun, ...) \
    DEFINE_SIMD(__CV_CAT(fun, 16s), short, v_int16,  __VA_ARGS__)

#define DEFINE_SIMD_S32(fun, ...) \
    DEFINE_SIMD(__CV_CAT(fun, 32s), int, v_int32,  __VA_ARGS__)

#define DEFINE_SIMD_F32(fun, ...) \
    DEFINE_SIMD(__CV_CAT(fun, 32f), float, v_float32, __VA_ARGS__)

#if CV_SIMD_64F
    #define DEFINE_SIMD_F64(fun, ...) \
        DEFINE_SIMD(__CV_CAT(fun, 64f), double, v_float64, __VA_ARGS__)
#else
    #define DEFINE_SIMD_F64(fun, ...) \
        DEFINE_NOSIMD(__CV_CAT(fun, 64f), double, __VA_ARGS__)
#endif

#define DEFINE_SIMD_SAT(fun, ...)      \
    DEFINE_SIMD_U8(fun, __VA_ARGS__)   \
    DEFINE_SIMD_S8(fun, __VA_ARGS__)   \
    DEFINE_SIMD_U16(fun, __VA_ARGS__)  \
    DEFINE_SIMD_S16(fun, __VA_ARGS__)

#define DEFINE_SIMD_NSAT(fun, ...)     \
    DEFINE_SIMD_S32(fun, __VA_ARGS__)  \
    DEFINE_SIMD_F32(fun, __VA_ARGS__)  \
    DEFINE_SIMD_F64(fun, __VA_ARGS__)

#define DEFINE_SIMD_ALL(fun, ...)      \
    DEFINE_SIMD_SAT(fun, __VA_ARGS__)  \
    DEFINE_SIMD_NSAT(fun, __VA_ARGS__)

#endif // SIMD_GUARD

///////////////////////////////////////////////////////////////////////////

namespace cv { namespace hal {

#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
#endif

#ifdef ARITHM_DEFINITIONS_ONLY

//=======================================
// Arithmetic and logical operations
// +, -, *, /, &, |, ^, ~, abs ...
//=======================================

///////////////////////////// Operations //////////////////////////////////

// Add
template<typename T1, typename Tvec>
struct op_add
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a + b; }
    static inline T1 r(T1 a, T1 b)
    { return a + b; }
};
template<typename T1, typename Tvec>
struct op_adds
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a + b; }
    static inline T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(a + b); }
};


// Subtract
template<typename T1, typename Tvec>
struct op_sub
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a - b; }
    static inline T1 r(T1 a, T1 b)
    { return a - b; }
};
template<typename T1, typename Tvec>
struct op_subs
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a - b; }
    static inline T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(a - b); }
};


// Max & Min
template<typename T1, typename Tvec>
struct op_max
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return v_max(a, b); }
    static inline T1 r(T1 a, T1 b)
    { return std::max(a, b); }
};
template<typename T1, typename Tvec>
struct op_min
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return v_min(a, b); }
    static inline T1 r(T1 a, T1 b)
    { return std::min(a, b); }
};

// Absolute difference
template<typename T1, typename Tvec>
struct op_absdiff
{
    Tvec r(const Tvec& a, const Tvec& b);
    T1 r(T1 a, T1 b);
};
// 'specializations to prevent "-0" results'
template<typename Tvec>
struct op_absdiff<float, Tvec>
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static inline float r(float a, float b)
    { return std::abs(a - b); }
};
template<typename Tvec>
struct op_absdiff<double, Tvec>
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static inline double r(double a, double b)
    { return std::abs(a - b); }
};
template<>
struct op_absdiff<int, v_int32>
{
    static inline v_int32 r(const v_int32& a, const v_int32& b)
    { return v_reinterpret_as_s32(v_absdiff(a, b)); }
    static inline int r(int a, int b)
    { return a > b ? a - b : b - a; }
};
template<typename T1, typename Tvec>
struct op_absdiff_u
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiff(a, b); }
    static inline T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(std::abs(a - b)); }
};
template<typename T1, typename Tvec>
struct op_absdiff_s
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return v_absdiffs(a, b); }
    static inline T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(std::abs(a - b)); }
};


// Logical
template<typename T1, typename Tvec>
struct op_or
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a | b; }
    static inline T1 r(T1 a, T1 b)
    { return a | b; }
};
template<typename T1, typename Tvec>
struct op_xor
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a ^ b; }
    static inline T1 r(T1 a, T1 b)
    { return a ^ b; }
};
template<typename T1, typename Tvec>
struct op_and
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a & b; }
    static inline T1 r(T1 a, T1 b)
    { return a & b; }
};
template<typename T1, typename Tvec>
struct op_not
{
    // ignored b from loader level
    static inline Tvec r(const Tvec& a)
    { return ~a; }
    static inline T1 r(T1 a, T1)
    { return ~a; }
};

//////////////////////////// Loaders /////////////////////////////////

#if CV_SIMD

template< template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct bin_loader
{
    typedef OP<T1, Tvec> op;

    static inline void l(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec a = vx_load(src1);
        Tvec b = vx_load(src2);
        v_store(dst, op::r(a, b));
    }

    static inline void la(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec a = vx_load_aligned(src1);
        Tvec b = vx_load_aligned(src2);
        v_store_aligned(dst, op::r(a, b)); // todo: try write without cache
    }

    static inline void l64(const T1* src1, const T1* src2, T1* dst)
    {
        Tvec a = vx_load_low(src1), b = vx_load_low(src2);
        v_store_low(dst, op::r(a, b));
    }
};

// void src2 for operation "not"
template<typename T1, typename Tvec>
struct bin_loader<op_not, T1, Tvec>
{
    typedef op_not<T1, Tvec> op;

    static inline void l(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load(src1);
        v_store(dst, op::r(a));
    }

    static inline void la(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load_aligned(src1);
        v_store_aligned(dst, op::r(a));
    }

    static inline void l64(const T1* src1, const T1*, T1* dst)
    {
        Tvec a = vx_load_low(src1);
        v_store_low(dst, op::r(a));
    }
};

#endif // CV_SIMD

//////////////////////////// Loops /////////////////////////////////

template<typename T1, typename T2>
static inline bool is_aligned(const T1* src1, const T1* src2, const T2* dst)
{ return (((size_t)src1|(size_t)src2|(size_t)dst) & (CV_SIMD_WIDTH - 1)) == 0; }

template< template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
static void bin_loop(const T1* src1, size_t step1, const T1* src2, size_t step2, T1* dst, size_t step, int width, int height)
{
    typedef OP<T1, Tvec> op;
#if CV_SIMD
    typedef bin_loader<OP, T1, Tvec> ldr;
    const int wide_step = Tvec::nlanes;
    #if !CV_NEON && CV_SIMD_WIDTH == 16
        const int wide_step_l = wide_step * 2;
    #else
        const int wide_step_l = wide_step;
    #endif
#endif // CV_SIMD

    for (; height--;
        src1 = (const T1 *)((const uchar *)src1 + step1),
        src2 = (const T1 *)((const uchar *)src2 + step2),
        dst  = (T1 *)((uchar *)dst + step)
    )
    {
        int x = 0;

    #if CV_SIMD
        #if !CV_NEON
        if (is_aligned(src1, src2, dst))
        {
            for (; x <= width - wide_step_l; x += wide_step_l)
            {
                ldr::la(src1 + x, src2 + x, dst + x);
                #if !CV_NEON && CV_SIMD_WIDTH == 16
                ldr::la(src1 + x + wide_step, src2 + x + wide_step, dst + x + wide_step);
                #endif
            }
        }
        else
        #endif
            for (; x <= width - wide_step_l; x += wide_step_l)
            {
                ldr::l(src1 + x, src2 + x, dst + x);
                #if !CV_NEON && CV_SIMD_WIDTH == 16
                ldr::l(src1 + x + wide_step, src2 + x + wide_step, dst + x + wide_step);
                #endif
            }

        #if CV_SIMD_WIDTH == 16
        for (; x <= width - 8/(int)sizeof(T1); x += 8/(int)sizeof(T1))
        {
            ldr::l64(src1 + x, src2 + x, dst + x);
        }
        #endif
    #endif // CV_SIMD

    #if CV_ENABLE_UNROLLED || CV_SIMD_WIDTH > 16
        while (x <= width - 4)
        {
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
        }
    #endif

        for (; x < width; x++)
            dst[x] = op::r(src1[x], src2[x]);
    }

    vx_cleanup();
}

#if !CV_SIMD_64F
template< template<typename T1, typename Tvec> class OP, typename T1>
static void bin_loop_nosimd(const T1* src1, size_t step1, const T1* src2, size_t step2, T1* dst, size_t step, int width, int height)
{
    typedef OP<T1, v_int32 /*dummy*/> op;

    for (; height--;
        src1 = (const T1 *)((const uchar *)src1 + step1),
        src2 = (const T1 *)((const uchar *)src2 + step2),
        dst  = (T1 *)((uchar *)dst + step)
    )
    {
        int x = 0;

        while (x <= width - 4)
        {
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
        }

        for (; x < width; x++)
            dst[x] = op::r(src1[x], src2[x]);
    }
}
#endif //!CV_SIMD_64F

#endif // ARITHM_DEFINITIONS_ONLY

////////////////////////////////////////////////////////////////////////////////////

#ifndef SIMD_GUARD
#define BIN_ARGS(_T1) const _T1* src1, size_t step1, const _T1* src2, size_t step2, \
                      _T1* dst, size_t step, int width, int height

#define BIN_ARGS_PASS src1, step1, src2, step2, dst, step, width, height
#endif // SIMD_GUARD

#undef DECLARE_SIMD_FUN
#define DECLARE_SIMD_FUN(fun, _T1) void fun(BIN_ARGS(_T1));

#undef DISPATCH_SIMD_FUN
#define DISPATCH_SIMD_FUN(fun, _T1, _Tvec, _OP)                           \
    void fun(BIN_ARGS(_T1), void*)                                        \
    {                                                                     \
        CV_INSTRUMENT_REGION();                                           \
        CALL_HAL(fun, __CV_CAT(cv_hal_, fun), BIN_ARGS_PASS)              \
        ARITHM_CALL_IPP(__CV_CAT(arithm_ipp_, fun), BIN_ARGS_PASS)        \
        CV_CPU_DISPATCH(fun, (BIN_ARGS_PASS), CV_CPU_DISPATCH_MODES_ALL); \
    }

#undef DEFINE_SIMD_FUN
#define DEFINE_SIMD_FUN(fun, _T1, _Tvec, _OP)     \
    void fun(BIN_ARGS(_T1))                       \
    {                                             \
        CV_INSTRUMENT_REGION();                   \
        bin_loop<_OP, _T1, _Tvec>(BIN_ARGS_PASS); \
    }

#undef DEFINE_NOSIMD_FUN
#define DEFINE_NOSIMD_FUN(fun, _T1, _OP)          \
    void fun(BIN_ARGS(_T1))                       \
    {                                             \
        CV_INSTRUMENT_REGION();                   \
        bin_loop_nosimd<_OP, _T1>(BIN_ARGS_PASS); \
    }

DEFINE_SIMD_SAT(add, op_adds)
DEFINE_SIMD_NSAT(add, op_add)

DEFINE_SIMD_SAT(sub, op_subs)
DEFINE_SIMD_NSAT(sub, op_sub)

DEFINE_SIMD_ALL(min, op_min)
DEFINE_SIMD_ALL(max, op_max)

DEFINE_SIMD_U8(absdiff, op_absdiff_u)
DEFINE_SIMD_S16(absdiff, op_absdiff_s)
DEFINE_SIMD_U16(absdiff, op_absdiff_u)
DEFINE_SIMD_S8(absdiff, op_absdiff_s)
DEFINE_SIMD_NSAT(absdiff, op_absdiff)

DEFINE_SIMD_U8(or,  op_or)
DEFINE_SIMD_U8(xor, op_xor)
DEFINE_SIMD_U8(and, op_and)

// One source!, an exception for operation "not"
// we could use macros here but it's better to implement it
// with that way to give more clarification
// about how macroS "DEFINE_SIMD_*" are works

#if defined(ARITHM_DECLARATIONS_ONLY) || defined(ARITHM_DEFINITIONS_ONLY)
void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height);
#endif
#ifdef ARITHM_DEFINITIONS_ONLY
void not8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    CV_INSTRUMENT_REGION();
    bin_loop<op_not, uchar, v_uint8>(src1, step1, src2, step2, dst, step, width, height);
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

//=======================================
// Compare
//=======================================

#ifdef ARITHM_DEFINITIONS_ONLY

///////////////////////////// Operations //////////////////////////////////

template<typename T1, typename Tvec>
struct op_cmplt
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a < b; }
    static inline uchar r(T1 a, T1 b)
    { return (uchar)-(int)(a < b); }
};

template<typename T1, typename Tvec>
struct op_cmple
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a <= b; }
    static inline uchar r(T1 a, T1 b)
    { return (uchar)-(int)(a <= b); }
};

template<typename T1, typename Tvec>
struct op_cmpeq
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a == b; }
    static inline uchar r(T1 a, T1 b)
    { return (uchar)-(int)(a == b); }
};

template<typename T1, typename Tvec>
struct op_cmpne
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a != b; }
    static inline uchar r(T1 a, T1 b)
    { return (uchar)-(int)(a != b); }
};

//////////////////////////// Loaders /////////////////////////////////

#if CV_SIMD
// todo: add support for RW alignment & stream
template<int nload, template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct cmp_loader_n
{
    void l(const T1* src1, const T1* src2, uchar* dst);
};

template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct cmp_loader_n<sizeof(uchar), OP, T1, Tvec>
{
    typedef OP<T1, Tvec> op;

    static inline void l(const T1* src1, const T1* src2, uchar* dst)
    {
        Tvec a = vx_load(src1);
        Tvec b = vx_load(src2);
        v_store(dst, v_reinterpret_as_u8(op::r(a, b)));
    }
};

template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct cmp_loader_n<sizeof(ushort), OP, T1, Tvec>
{
    typedef OP<T1, Tvec> op;
    enum {step = Tvec::nlanes};

    static inline void l(const T1* src1, const T1* src2, uchar* dst)
    {
        Tvec c0 = op::r(vx_load(src1), vx_load(src2));
        Tvec c1 = op::r(vx_load(src1 + step), vx_load(src2 + step));
        v_store(dst, v_pack_b(v_reinterpret_as_u16(c0), v_reinterpret_as_u16(c1)));
    }
};

template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct cmp_loader_n<sizeof(unsigned), OP, T1, Tvec>
{
    typedef OP<T1, Tvec> op;
    enum {step = Tvec::nlanes};

    static inline void l(const T1* src1, const T1* src2, uchar* dst)
    {
        v_uint32 c0 = v_reinterpret_as_u32(op::r(vx_load(src1), vx_load(src2)));
        v_uint32 c1 = v_reinterpret_as_u32(op::r(vx_load(src1 + step), vx_load(src2 + step)));
        v_uint32 c2 = v_reinterpret_as_u32(op::r(vx_load(src1 + step * 2), vx_load(src2 + step * 2)));
        v_uint32 c3 = v_reinterpret_as_u32(op::r(vx_load(src1 + step * 3), vx_load(src2 + step * 3)));
        v_store(dst, v_pack_b(c0, c1, c2, c3));
    }
};

template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct cmp_loader_n<sizeof(double), OP, T1, Tvec>
{
    typedef OP<T1, Tvec> op;
    enum {step = Tvec::nlanes};

    static inline void l(const T1* src1, const T1* src2, uchar* dst)
    {
        v_uint64 c0 = v_reinterpret_as_u64(op::r(vx_load(src1), vx_load(src2)));
        v_uint64 c1 = v_reinterpret_as_u64(op::r(vx_load(src1 + step), vx_load(src2 + step)));
        v_uint64 c2 = v_reinterpret_as_u64(op::r(vx_load(src1 + step * 2), vx_load(src2 + step * 2)));
        v_uint64 c3 = v_reinterpret_as_u64(op::r(vx_load(src1 + step * 3), vx_load(src2 + step * 3)));

        v_uint64 c4 = v_reinterpret_as_u64(op::r(vx_load(src1 + step * 4), vx_load(src2 + step * 4)));
        v_uint64 c5 = v_reinterpret_as_u64(op::r(vx_load(src1 + step * 5), vx_load(src2 + step * 5)));
        v_uint64 c6 = v_reinterpret_as_u64(op::r(vx_load(src1 + step * 6), vx_load(src2 + step * 6)));
        v_uint64 c7 = v_reinterpret_as_u64(op::r(vx_load(src1 + step * 7), vx_load(src2 + step * 7)));
        v_store(dst, v_pack_b(c0, c1, c2, c3, c4, c5, c6, c7));
    }
};

#endif // CV_SIMD

//////////////////////////// Loops /////////////////////////////////

template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
static void cmp_loop(const T1* src1, size_t step1, const T1* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    typedef OP<T1, Tvec> op;
#if CV_SIMD
    typedef cmp_loader_n<sizeof(T1), OP, T1, Tvec> ldr;
    const int wide_step = Tvec::nlanes * sizeof(T1);
#endif // CV_SIMD

    for (; height--;
        src1 = (const T1 *)((const uchar *)src1 + step1),
        src2 = (const T1 *)((const uchar *)src2 + step2),
        dst  += step
    )
    {
        int x = 0;

    #if CV_SIMD
        for (; x <= width - wide_step; x += wide_step)
        {
            ldr::l(src1 + x, src2 + x, dst + x);
        }
    #endif // CV_SIMD

    #if CV_ENABLE_UNROLLED || CV_SIMD_WIDTH > 16
        while (x <= width - 4)
        {
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
        }
    #endif

        for (; x < width; x++)
            dst[x] = op::r(src1[x], src2[x]);
    }

    vx_cleanup();
}

template<typename T1, typename Tvec>
static void cmp_loop(const T1* src1, size_t step1, const T1* src2, size_t step2,
                     uchar* dst, size_t step, int width, int height, int cmpop)
{
    switch(cmpop)
    {
    case CMP_LT:
        cmp_loop<op_cmplt, T1, Tvec>(src1, step1, src2, step2, dst, step, width, height);
        break;
    case CMP_GT:
        cmp_loop<op_cmplt, T1, Tvec>(src2, step2, src1, step1, dst, step, width, height);
        break;
    case CMP_LE:
        cmp_loop<op_cmple, T1, Tvec>(src1, step1, src2, step2, dst, step, width, height);
        break;
    case CMP_GE:
        cmp_loop<op_cmple, T1, Tvec>(src2, step2, src1, step1, dst, step, width, height);
        break;
    case CMP_EQ:
        cmp_loop<op_cmpeq, T1, Tvec>(src1, step1, src2, step2, dst, step, width, height);
        break;
    default:
        CV_Assert(cmpop == CMP_NE);
        cmp_loop<op_cmpne, T1, Tvec>(src1, step1, src2, step2, dst, step, width, height);
        break;
    }
}

#if !CV_SIMD_64F
template< template<typename T1, typename Tvec> class OP, typename T1>
static void cmp_loop_nosimd(const T1* src1, size_t step1, const T1* src2, size_t step2, uchar* dst, size_t step, int width, int height)
{
    typedef OP<T1, v_int32 /*dummy*/> op;

    for (; height--;
        src1 = (const T1 *)((const uchar *)src1 + step1),
        src2 = (const T1 *)((const uchar *)src2 + step2),
        dst += step
    )
    {
        int x = 0;

        while (x <= width - 4)
        {
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
            dst[x] = op::r(src1[x], src2[x]); x++;
        }

        for (; x < width; x++)
            dst[x] = op::r(src1[x], src2[x]);
    }
}
static void cmp_loop_nosimd(const double* src1, size_t step1, const double* src2, size_t step2,
                            uchar* dst, size_t step, int width, int height, int cmpop)
{
    switch(cmpop)
    {
    case CMP_LT:
        cmp_loop_nosimd<op_cmplt, double>(src1, step1, src2, step2, dst, step, width, height);
        break;
    case CMP_GT:
        cmp_loop_nosimd<op_cmplt, double>(src2, step2, src1, step1, dst, step, width, height);
        break;
    case CMP_LE:
        cmp_loop_nosimd<op_cmple, double>(src1, step1, src2, step2, dst, step, width, height);
        break;
    case CMP_GE:
        cmp_loop_nosimd<op_cmple, double>(src2, step2, src1, step1, dst, step, width, height);
        break;
    case CMP_EQ:
        cmp_loop_nosimd<op_cmpeq, double>(src1, step1, src2, step2, dst, step, width, height);
        break;
    default:
        CV_Assert(cmpop == CMP_NE);
        cmp_loop_nosimd<op_cmpne, double>(src1, step1, src2, step2, dst, step, width, height);
        break;
    }
}
#endif // !CV_SIMD_64F

#endif // ARITHM_DEFINITIONS_ONLY

/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef SIMD_GUARD
#define CMP_ARGS(_T1) const _T1* src1, size_t step1, const _T1* src2, size_t step2, \
                           uchar* dst, size_t step, int width, int height

#define CMP_ARGS_PASS src1, step1, src2, step2, dst, step, width, height
#endif // SIMD_GUARD

#undef DECLARE_SIMD_FUN
#define DECLARE_SIMD_FUN(fun, _T1) void fun(CMP_ARGS(_T1), int cmpop);

#undef DISPATCH_SIMD_FUN
#define DISPATCH_SIMD_FUN(fun, _T1, _Tvec, ...)                                          \
    void fun(CMP_ARGS(_T1), void* _cmpop)                                                \
    {                                                                                    \
        CV_INSTRUMENT_REGION();                                                          \
        CALL_HAL(fun, __CV_CAT(cv_hal_, fun), CMP_ARGS_PASS, *(int*)_cmpop)              \
        ARITHM_CALL_IPP(__CV_CAT(arithm_ipp_, fun), CMP_ARGS_PASS, *(int*)_cmpop)        \
        CV_CPU_DISPATCH(fun, (CMP_ARGS_PASS, *(int*)_cmpop), CV_CPU_DISPATCH_MODES_ALL); \
    }

#undef DEFINE_SIMD_FUN
#define DEFINE_SIMD_FUN(fun, _T1, _Tvec, ...)       \
    void fun(CMP_ARGS(_T1), int cmpop)              \
    {                                               \
        CV_INSTRUMENT_REGION();                     \
        cmp_loop<_T1, _Tvec>(CMP_ARGS_PASS, cmpop); \
    }

#undef DEFINE_NOSIMD_FUN
#define DEFINE_NOSIMD_FUN(fun, _T1, _Tvec, ...)     \
    void fun(CMP_ARGS(_T1), int cmpop)              \
    {                                               \
        CV_INSTRUMENT_REGION();                     \
        cmp_loop_nosimd(CMP_ARGS_PASS, cmpop);      \
    }

// todo: try to avoid define dispatcher functions using macros with these such cases
DEFINE_SIMD_ALL(cmp)

//=========================================================================
// Dual source scaling helpers
//
// Multiply, Div, AddWeighted
//=========================================================================

#ifdef ARITHM_DEFINITIONS_ONLY

//////////////////////////// Loaders ///////////////////////////////

#if CV_SIMD
// todo: add support for RW alignment & stream
template<int nload, template<typename T1, typename T2, typename Tvec> class OP, typename T1, typename T2, typename Tvec>
struct scalar_loader_n
{
    void l(const T1* src1, const T1* src2, const T2* scalar, T1* dst);
};

template<template<typename T1, typename T2, typename Tvec> class OP, typename T1, typename T2, typename Tvec>
struct scalar_loader_n<sizeof(uchar), OP, T1, T2, Tvec>
{
    typedef typename V_RegTraits<Tvec>::w_reg Twvec;
    typedef typename V_RegTraits<Tvec>::q_reg Tqvec;
    typedef OP<T1, T2, Twvec> op;

    static inline void l(const T1* src1, const T1* src2, const T2* scalar, T1* dst)
    {
        Twvec v_src1 = vx_load_expand(src1);
        Twvec v_src2 = vx_load_expand(src2);

        Tqvec t0, t1, t2, t3;
        v_expand(v_src1, t0, t1);
        v_expand(v_src2, t2, t3);

        v_int32 r0 = op::r(v_reinterpret_as_s32(t0), v_reinterpret_as_s32(t2), scalar);
        v_int32 r1 = op::r(v_reinterpret_as_s32(t1), v_reinterpret_as_s32(t3), scalar);

        store(dst, v_src2, r0, r1);
    }

    static inline void store(uchar* dst, const Twvec& src, const v_int32& a, const v_int32& b)
    {
        v_pack_store(dst, op::pre(src, v_pack_u(a, b)));
    }
    static inline void store(schar* dst, const Twvec& src, const v_int32& a, const v_int32& b)
    {
        v_pack_store(dst, op::pre(src, v_pack(a, b)));
    }
};

template<template<typename T1, typename T2, typename Tvec> class OP, typename T1, typename T2, typename Tvec>
struct scalar_loader_n<sizeof(ushort), OP, T1, T2, Tvec>
{
    typedef typename V_RegTraits<Tvec>::w_reg Twvec;
    typedef OP<T1, T2, Tvec> op;

    static inline void l(const T1* src1, const T1* src2, const T2* scalar, T1* dst)
    {
        Tvec v_src1 = vx_load(src1);
        Tvec v_src2 = vx_load(src2);

        Twvec t0, t1, t2, t3;
        v_expand(v_src1, t0, t1);
        v_expand(v_src2, t2, t3);

        v_int32 r0 = op::r(v_reinterpret_as_s32(t0), v_reinterpret_as_s32(t2), scalar);
        v_int32 r1 = op::r(v_reinterpret_as_s32(t1), v_reinterpret_as_s32(t3), scalar);

        store(dst, v_src2, r0, r1);
    }

    static inline void store(ushort* dst, const Tvec& src, const v_int32& a, const v_int32& b)
    {
        v_store(dst, op::pre(src, v_pack_u(a, b)));
    }
    static inline void store(short* dst, const Tvec& src, const v_int32& a, const v_int32& b)
    {
        v_store(dst, op::pre(src, v_pack(a, b)));
    }
};

template<template<typename T1, typename T2, typename Tvec> class OP, typename T2>
struct scalar_loader_n<sizeof(int), OP, int, T2, v_int32>
{
    typedef OP<int, T2, v_int32> op;
    enum {step = v_int32::nlanes};

    static inline void l(const int* src1, const int* src2, const T2* scalar, int* dst)
    {
        v_int32 v_src1 = vx_load(src1);
        v_int32 v_src2 = vx_load(src2);
        v_int32 r0 = op::r(v_src1, v_src2, scalar);
                r0 = op::pre(v_src2, r0);

        v_src1 = vx_load(src1 + step);
        v_src2 = vx_load(src2 + step);
        v_int32 r1 = op::r(v_src1, v_src2, scalar);
                r1 = op::pre(v_src2, r1);

        v_store(dst, r0);
        v_store(dst + step, r1);
    }
};

template<template<typename T1, typename T2, typename Tvec> class OP>
struct scalar_loader_n<sizeof(int), OP, int, double, v_int32>
{
    typedef OP<int, float, v_int32> op;
    typedef OP<double, double, v_float64> op64;
    enum {step = v_int32::nlanes};

    static inline void l(const int* src1, const int* src2, const double* scalar, int* dst)
    {
        v_int32 v_src1 = vx_load(src1);
        v_int32 v_src2 = vx_load(src2);
        v_int32 r0 = r(v_src1, v_src2, scalar);
                r0 = op::pre(v_src2, r0);

        v_src1 = vx_load(src1 + step);
        v_src2 = vx_load(src2 + step);
        v_int32 r1 = r(v_src1, v_src2, scalar);
                r1 = op::pre(v_src2, r1);

        v_store(dst, r0);
        v_store(dst + step, r1);
    }

    static inline v_int32 r(const v_int32& a, const v_int32& b, const double* scalar)
    {
        v_float64 f0, f1, f2, f3;
        f0 = v_cvt_f64(a);
        f1 = v_cvt_f64_high(a);
        f2 = v_cvt_f64(b);
        f3 = v_cvt_f64_high(b);

        v_float64 r0 = op64::r(f0, f2, scalar);
        v_float64 r1 = op64::r(f1, f3, scalar);

        return v_round(r0, r1);
    }
};

template<template<typename T1, typename T2, typename Tvec> class OP, typename T2>
struct scalar_loader_n<sizeof(float), OP, float, T2, v_float32>
{
    typedef OP<float, T2, v_float32> op;
    enum {step = v_float32::nlanes};

    static inline void l(const float* src1, const float* src2, const T2* scalar, float* dst)
    {
        v_float32 f0 = vx_load(src1);
        v_float32 f1 = vx_load(src2);
        v_float32 f2 = vx_load(src1 + step);
        v_float32 f3 = vx_load(src2 + step);

        f0 = op::r(f0, f1, scalar);
        f0 = op::pre(f1, f0);

        f2 = op::r(f2, f3, scalar);
        f2 = op::pre(f3, f2);

        v_store(dst, f0);
        v_store(dst + step, f2);
    }
};

template<template<typename T1, typename T2, typename Tvec> class OP>
struct scalar_loader_n<sizeof(float), OP, float, double, v_float32>
{
    typedef OP<float, float, v_float32> op;
    typedef OP<double, double, v_float64> op64;
    enum {step = v_float32::nlanes};

    static inline void l(const float* src1, const float* src2, const double* scalar, float* dst)
    {
        v_float32 f0 = vx_load(src1);
        v_float32 f1 = vx_load(src2);
        v_float32 f2 = vx_load(src1 + step);
        v_float32 f3 = vx_load(src2 + step);

        f0 = r(f0, f1, scalar);
        f0 = op::pre(f1, f0);

        f2 = r(f2, f3, scalar);
        f2 = op::pre(f3, f2);

        v_store(dst, f0);
        v_store(dst + step, f2);
    }
    static inline v_float32 r(const v_float32& a, const v_float32& b, const double* scalar)
    {
        v_float64 f0, f1, f2, f3;
        f0 = v_cvt_f64(a);
        f1 = v_cvt_f64_high(a);
        f2 = v_cvt_f64(b);
        f3 = v_cvt_f64_high(b);

        v_float64 r0 = op64::r(f0, f2, scalar);
        v_float64 r1 = op64::r(f1, f3, scalar);

        return v_cvt_f32(r0, r1);
    }
};
#endif // CV_SIMD

#if CV_SIMD_64F
template<template<typename T1, typename T2, typename Tvec> class OP>
struct scalar_loader_n<sizeof(double), OP, double, double, v_float64>
{
    typedef OP<double, double, v_float64> op;
    enum {step = v_float64::nlanes};

    static inline void l(const double* src1, const double* src2, const double* scalar, double* dst)
    {
        v_float64 f0 = vx_load(src1);
        v_float64 f1 = vx_load(src2);
        v_float64 f2 = vx_load(src1 + step);
        v_float64 f3 = vx_load(src2 + step);

        f0 = op::r(f0, f1, scalar);
        f0 = op::pre(f1, f0);

        f2 = op::r(f2, f3, scalar);
        f2 = op::pre(f3, f2);

        v_store(dst, f0);
        v_store(dst + step, f2);
    }
};
#endif // CV_SIMD_64F

//////////////////////////// Loops /////////////////////////////////

template<template<typename T1, typename T2, typename Tvec> class OP, typename T1, typename T2, typename Tvec>
void scalar_loop(const T1* src1, size_t step1, const T1* src2, size_t step2,
                 T1* dst, size_t step, int width, int height, const T2* scalar)
{
    typedef OP<T1, T2, Tvec> op;
#if CV_SIMD
    typedef scalar_loader_n<sizeof(T1), OP, T1, T2, Tvec> ldr;
    const int wide_step = sizeof(T1) > sizeof(ushort) ? Tvec::nlanes * 2 :
                          sizeof(T1) == sizeof(uchar) ? Tvec::nlanes / 2 : Tvec::nlanes;
#endif // CV_SIMD

    step1 /= sizeof(T1);
    step2 /= sizeof(T1);
    step  /= sizeof(T1);

    for (; height--; src1 += step1, src2 += step2, dst += step)
    {
        int x = 0;

    #if CV_SIMD
        for (; x <= width - wide_step; x += wide_step)
        {
            ldr::l(src1 + x, src2 + x, scalar, dst + x);
        }
    #endif // CV_SIMD

    #if CV_ENABLE_UNROLLED || CV_SIMD_WIDTH > 16
        for (; x <= width - 4; x += 4)
        {
            T1 t0 = op::r(src1[x], src2[x], scalar);
            T1 t1 = op::r(src1[x + 1], src2[x + 1], scalar);
            dst[x] = t0; dst[x + 1] = t1;

            t0 = op::r(src1[x + 2], src2[x + 2], scalar);
            t1 = op::r(src1[x + 3], src2[x + 3], scalar);
            dst[x + 2] = t0; dst[x + 3] = t1;
        }
    #endif

        for (; x < width; ++x)
            dst[x] = op::r(src1[x], src2[x], scalar);
    }

    vx_cleanup();
}

#if !CV_SIMD_64F
template<template<typename T1, typename T2, typename Tvec> class OP, typename T1, typename T2, typename Tvec>
void scalar_loop_nosimd(const T1* src1, size_t step1, const T1* src2, size_t step2,
                 T1* dst, size_t step, int width, int height, const T2* scalar)
{
    typedef OP<T1, T2, Tvec> op;

    step1 /= sizeof(T1);
    step2 /= sizeof(T1);
    step  /= sizeof(T1);

    for (; height--; src1 += step1, src2 += step2, dst += step)
    {
        int x = 0;

        for (; x <= width - 4; x += 4)
        {
            T1 t0 = op::r(src1[x], src2[x], scalar);
            T1 t1 = op::r(src1[x + 1], src2[x + 1], scalar);
            dst[x] = t0; dst[x + 1] = t1;

            t0 = op::r(src1[x + 2], src2[x + 2], scalar);
            t1 = op::r(src1[x + 3], src2[x + 3], scalar);
            dst[x + 2] = t0; dst[x + 3] = t1;
        }

        for (; x < width; ++x)
            dst[x] = op::r(src1[x], src2[x], scalar);
    }
}
#define SCALAR_LOOP64F scalar_loop_nosimd
#else
#define SCALAR_LOOP64F scalar_loop
#endif // !CV_SIMD_64F

#endif // ARITHM_DEFINITIONS_ONLY

//=========================================================================
// Multiply
//=========================================================================

#ifdef ARITHM_DEFINITIONS_ONLY

///////////////////////////// Operations //////////////////////////////////

template<typename T1, typename Tvec>
struct op_mul
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a * b; }
    static inline T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(a * b); }
};

template<typename T1, typename T2, typename Tvec>
struct op_mul_scale
{
    static inline v_int32 r(const v_int32& a, const v_int32& b, const T2* scalar)
    {
        const v_float32 v_scalar = vx_setall_f32(*scalar);
        v_float32 f0 = v_cvt_f32(a);
        v_float32 f1 = v_cvt_f32(b);
        return v_round(v_scalar * f0 * f1);
    }
    static inline T1 r(T1 a, T1 b, const T2* scalar)
    { return saturate_cast<T1>((*scalar) * (T2)a * b); }
    static inline Tvec pre(const Tvec&, const Tvec& res)
    { return res; }
};

template<>
struct op_mul_scale<float, float, v_float32>
{
    static inline v_float32 r(const v_float32& a, const v_float32& b, const float* scalar)
    {
        const v_float32 v_scalar = vx_setall_f32(*scalar);
        return v_scalar * a * b;
    }
    static inline float r(float a, float b, const float* scalar)
    { return (*scalar) * a * b; }
    static inline v_float32 pre(const v_float32&, const v_float32& res)
    { return res; }
};

template<>
struct op_mul_scale<double, double, v_float64>
{
#if CV_SIMD_64F
    static inline v_float64 r(const v_float64& a, const v_float64& b, const double* scalar)
    {
        const v_float64 v_scalar = vx_setall_f64(*scalar);
        return v_scalar * a * b;
    }
#endif
    static inline double r(double a, double b, const double* scalar)
    { return (*scalar) * a * b; }
    static inline v_float64 pre(const v_float64&, const v_float64& res)
    { return res; }
};

//////////////////////////// Loops /////////////////////////////////

template<typename T1, typename Tvec>
void mul_loop(const T1* src1, size_t step1, const T1* src2, size_t step2,
              T1* dst, size_t step, int width, int height, const double* scalar)
{
    /*
    if (*scalar == 1.0) {
        bin_loop<op_mul, T1, Tvec>(src1, step1, src2, step2, dst, step, width, height);
        return;
    }*/
    float fscalar = (float)*scalar;
    if (std::fabs(fscalar - 1) > FLT_EPSILON)
        scalar_loop<op_mul_scale, T1, double, Tvec>(src1, step1, src2, step2,
            dst, step, width, height, scalar);
    else
        scalar_loop<op_mul_scale, T1, float, Tvec>(src1, step1, src2, step2,
            dst, step, width, height, &fscalar);
}

template<>
void mul_loop<double, v_float64>(const double* src1, size_t step1, const double* src2, size_t step2,
                                double* dst, size_t step, int width, int height, const double* scalar)
{
    if (*scalar == 1.0) {
        bin_loop<op_mul, double, v_float64>(src1, step1, src2, step2, dst, step, width, height);
        return;
    }
    scalar_loop<op_mul_scale, double, double, v_float64>(src1, step1, src2, step2,
        dst, step, width, height, scalar);
}

#endif // ARITHM_DEFINITIONS_ONLY

//////////////////////////////////////////////////////////////////////////

#ifndef SIMD_GUARD
#define SCALAR_ARGS(_T1) const _T1* src1, size_t step1, const _T1* src2, size_t step2, \
                          _T1* dst, size_t step, int width, int height
#define SCALAR_ARGS_PASS src1, step1, src2, step2, dst, step, width, height
#endif // SIMD_GUARD

#undef DECLARE_SIMD_FUN
#define DECLARE_SIMD_FUN(fun, _T1) void fun(SCALAR_ARGS(_T1), const double* scalar);

#undef DISPATCH_SIMD_FUN
#define DISPATCH_SIMD_FUN(fun, _T1, _Tvec, ...)                          \
    void fun(SCALAR_ARGS(_T1), void* scalar)                             \
    {                                                                    \
        CV_INSTRUMENT_REGION();                                          \
        CALL_HAL(fun, __CV_CAT(cv_hal_, fun),                            \
            SCALAR_ARGS_PASS, *(const double*)scalar)                    \
        ARITHM_CALL_IPP(__CV_CAT(arithm_ipp_, fun),                      \
            SCALAR_ARGS_PASS, *(const double*)scalar)                    \
        CV_CPU_DISPATCH(fun, (SCALAR_ARGS_PASS, (const double*)scalar),  \
            CV_CPU_DISPATCH_MODES_ALL);                                  \
    }

#undef DEFINE_SIMD_FUN
#define DEFINE_SIMD_FUN(fun, _T1, _Tvec, op)           \
    void fun(SCALAR_ARGS(_T1), const double* scalar)   \
    {                                                  \
        CV_INSTRUMENT_REGION();                        \
        op<_T1, _Tvec>(SCALAR_ARGS_PASS, scalar);      \
    }

DEFINE_SIMD_SAT(mul, mul_loop)
DEFINE_SIMD_F32(mul, mul_loop)
DEFINE_SIMD_S32(mul, mul_loop)
DEFINE_SIMD_F64(mul, mul_loop)

//=========================================================================
// Div
//=========================================================================

#ifdef ARITHM_DEFINITIONS_ONLY

///////////////////////////// Operations //////////////////////////////////

template<typename T1, typename Tvec>
struct op_div
{
    static inline Tvec r(const Tvec& a, const Tvec& b)
    { return a / b; }
    static inline T1 r(T1 a, T1 b)
    { return saturate_cast<T1>(a / b); }
};

template<typename T1, typename T2, typename Tvec>
struct op_div_scale
{
    static inline v_int32 r(const v_int32& a, const v_int32& b, const T2* scalar)
    {
        const v_float32 v_scalar = vx_setall_f32(*scalar);
        v_float32 f0 = v_cvt_f32(a);
        v_float32 f1 = v_cvt_f32(b);
        return v_round(f0 * v_scalar / f1);
    }
    static inline Tvec pre(const Tvec& denom, const Tvec& res)
    {
        const Tvec v_zero = Tvec();
        return v_select(denom == v_zero, v_zero, res);
    }
    static inline T1 r(T1 a, T1 denom, const T2* scalar)
    { return denom != 0 ? saturate_cast<T1>(a * (*scalar) / denom) : (T1)0; }
};

template<>
struct op_div_scale<float, float, v_float32>
{
    static inline v_float32 r(const v_float32& a, const v_float32& b, const float* scalar)
    {
        const v_float32 v_scalar = vx_setall_f32(*scalar);
        return a * v_scalar / b;
    }
    static inline v_float32 pre(const v_float32& denom, const v_float32& res)
    {
        const v_float32 v_zero = vx_setzero_f32();
        return v_select(denom == v_zero, v_zero, res);
    }
    static inline float r(float a, float denom, const float* scalar)
    { return denom != 0 ? a * (*scalar) / denom : 0.0f; }
};

template<>
struct op_div_scale<double, double, v_float64>
{
#if CV_SIMD_64F
    static inline v_float64 r(const v_float64& a, const v_float64& b, const double* scalar)
    {
        const v_float64 v_scalar = vx_setall_f64(*scalar);
        return a * v_scalar / b;
    }
    static inline v_float64 pre(const v_float64& denom, const v_float64& res)
    {
        const v_float64 v_zero = vx_setzero_f64();
        return v_select(denom == v_zero, v_zero, res);
    }
#endif
    static inline double r(double a, double denom, const double* scalar)
    { return denom != 0 ? a * (*scalar) / denom : 0.0; }
};

//////////////////////////// Loops /////////////////////////////////

template<typename T1, typename Tvec>
void div_loop(const T1* src1, size_t step1, const T1* src2, size_t step2,
              T1* dst, size_t step, int width, int height, const double* scalar)
{
    /*if (*scalar == 1.0) {
        bin_loop<op_div, T1, Tvec>(src1, step1, src2, step2, dst, step, width, height);
        return;
    }*/
    float fscalar = (float)*scalar;
    scalar_loop<op_div_scale, T1, float, Tvec>(src1, step1, src2, step2,
        dst, step, width, height, &fscalar);
}

template<>
void div_loop<double, v_float64>(const double* src1, size_t step1, const double* src2, size_t step2,
                                double* dst, size_t step, int width, int height, const double* scalar)
{
    if (*scalar == 1.0) {
        bin_loop<op_div, double, v_float64>(src1, step1, src2, step2, dst, step, width, height);
        return;
    }
    SCALAR_LOOP64F<op_div_scale, double, double, v_float64>(src1, step1, src2, step2,
        dst, step, width, height, scalar);
}


#endif // ARITHM_DEFINITIONS_ONLY

//////////////////////////////////////////////////////////////////////////

DEFINE_SIMD_SAT(div, div_loop)
DEFINE_SIMD_S32(div, div_loop)
DEFINE_SIMD_F32(div, div_loop)
DEFINE_SIMD_F64(div, div_loop)

//=========================================================================
// AddWeighted
//=========================================================================

#ifdef ARITHM_DEFINITIONS_ONLY

///////////////////////////// Operations //////////////////////////////////

///// Add scale
template<typename T1, typename T2, typename Tvec>
struct op_add_scale
{
    static inline v_int32 r(const v_int32& a, const v_int32& b, const T2* scalar)
    {
        const v_float32 v_alpha = vx_setall_f32(*scalar);
        v_float32 f0 = v_cvt_f32(a);
        v_float32 f1 = v_cvt_f32(b);
        return v_round(v_fma(f0, v_alpha, f1));
    }
    static inline T1 r(T1 a, T1 b, const T2* scalar)
    { return saturate_cast<T1>((T2)a * (*scalar) + b); }
    static inline Tvec pre(const Tvec&, const Tvec& res)
    { return res; }
};

template<>
struct op_add_scale<float, float, v_float32>
{
    static inline v_float32 r(const v_float32& a, const v_float32& b, const float* scalar)
    {
        const v_float32 v_alpha = vx_setall_f32(*scalar);
        return v_fma(a, v_alpha, b);
    }
    static inline float r(float a, float b, const float* scalar)
    { return a * (*scalar) + b; }
    static inline v_float32 pre(const v_float32&, const v_float32& res)
    { return res; }
};

template<>
struct op_add_scale<double, double, v_float64>
{
#if CV_SIMD_64F
    static inline v_float64 r(const v_float64& a, const v_float64& b, const double* scalar)
    {
        const v_float64 v_alpha = vx_setall_f64(*scalar);
        return v_fma(a, v_alpha, b);
    }
#endif
    static inline double r(double a, double b, const double* scalar)
    { return a * (*scalar) + b; }
    static inline v_float64 pre(const v_float64&, const v_float64& res)
    { return res; }
};

///// Weighted sum
template<typename T1, typename T2, typename Tvec>
struct op_add_weighted
{
    static inline v_int32 r(const v_int32& a, const v_int32& b, const T2* scalars)
    {
        const v_float32 v_alpha = vx_setall_f32(scalars[0]);
        const v_float32 v_beta  = vx_setall_f32(scalars[1]);
        const v_float32 v_gamma = vx_setall_f32(scalars[2]);
        v_float32 f0 = v_cvt_f32(a);
        v_float32 f1 = v_cvt_f32(b);
        return v_round(v_fma(f0, v_alpha, v_fma(f1, v_beta, v_gamma)));
    }
    static inline T1 r(T1 a, T1 b, const T2* scalars)
    { return saturate_cast<T1>(a * scalars[0] + b * scalars[1] + scalars[2]); }
    static inline Tvec pre(const Tvec&, const Tvec& res)
    { return res; }
};

template<>
struct op_add_weighted<float, float, v_float32>
{
    static inline v_float32 r(const v_float32& a, const v_float32& b, const float* scalars)
    {
        const v_float32 v_alpha = vx_setall_f32(scalars[0]);
        const v_float32 v_beta  = vx_setall_f32(scalars[1]);
        const v_float32 v_gamma = vx_setall_f32(scalars[2]);
        return v_fma(a, v_alpha, v_fma(b, v_beta, v_gamma));
    }
    static inline float r(float a, float b, const float* scalars)
    { return a * scalars[0] + b * scalars[1] + scalars[2]; }
    static inline v_float32 pre(const v_float32&, const v_float32& res)
    { return res; }
};

template<>
struct op_add_weighted<double, double, v_float64>
{
#if CV_SIMD_64F
    static inline v_float64 r(const v_float64& a, const v_float64& b, const double* scalars)
    {
        const v_float64 v_alpha = vx_setall_f64(scalars[0]);
        const v_float64 v_beta  = vx_setall_f64(scalars[1]);
        const v_float64 v_gamma = vx_setall_f64(scalars[2]);
        return v_fma(a, v_alpha, v_fma(b, v_beta, v_gamma));
    }
#endif
    static inline double r(double a, double b, const double* scalars)
    { return a * scalars[0] + b * scalars[1] + scalars[2]; }
    static inline v_float64 pre(const v_float64&, const v_float64& res)
    { return res; }
};

//////////////////////////// Loops /////////////////////////////////

template<typename T1, typename Tvec>
void add_weighted_loop(const T1* src1, size_t step1, const T1* src2, size_t step2,
                       T1* dst, size_t step, int width, int height, const double* scalars)
{
    float fscalars[] = {(float)scalars[0], (float)scalars[1], (float)scalars[2]};
    if (fscalars[1] == 1.0 && fscalars[2] == 0)
    {
        scalar_loop<op_add_scale, T1, float, Tvec>(src1, step1, src2, step2,
            dst, step, width, height, fscalars);
    }
    else
    {
        scalar_loop<op_add_weighted, T1, float, Tvec>(src1, step1, src2, step2,
            dst, step, width, height, fscalars);
    }
}
template<typename T1, typename Tvec>
void add_weighted_loop_d(const T1* src1, size_t step1, const T1* src2, size_t step2,
                         T1* dst, size_t step, int width, int height, const double* scalars)
{
    bool alpha_flt = std::fabs(scalars[0] - 1) <= FLT_EPSILON;
    bool beta_flt = std::fabs(scalars[1] - 1) <= FLT_EPSILON;

    if (alpha_flt && beta_flt)
    {
        add_weighted_loop<T1, Tvec>(src1, step1, src2, step2, dst, step, width, height, scalars);
        return;
    }
    if (scalars[1] == 1.0 && scalars[2] == 0)
    {
        SCALAR_LOOP64F<op_add_scale, T1, double, Tvec>(src1, step1, src2, step2,
            dst, step, width, height, scalars);
    }
    else
    {
        SCALAR_LOOP64F<op_add_weighted, T1, double, Tvec>(src1, step1, src2, step2,
            dst, step, width, height, scalars);
    }
}
template<>
void add_weighted_loop_d<double, v_float64>(const double* src1, size_t step1, const double* src2, size_t step2,
                                            double* dst, size_t step, int width, int height, const double* scalars)
{
    if (scalars[1] == 1.0 && scalars[2] == 0)
    {
        SCALAR_LOOP64F<op_add_scale, double, double, v_float64>(src1, step1, src2, step2,
            dst, step, width, height, scalars);
    }
    else
        SCALAR_LOOP64F<op_add_weighted, double, double, v_float64>(src1, step1, src2, step2,
            dst, step, width, height, scalars);
}

#endif // ARITHM_DEFINITIONS_ONLY

//////////////////////////////////////////////////////////////////////////

#undef DISPATCH_SIMD_FUN
#define DISPATCH_SIMD_FUN(fun, _T1, _Tvec, ...)                          \
    void fun(SCALAR_ARGS(_T1), void* scalar)                             \
    {                                                                    \
        CV_INSTRUMENT_REGION();                                          \
        CALL_HAL(fun, __CV_CAT(cv_hal_, fun),                            \
            SCALAR_ARGS_PASS, (const double*)scalar)                     \
        ARITHM_CALL_IPP(__CV_CAT(arithm_ipp_, fun),                      \
            SCALAR_ARGS_PASS, (const double*)scalar)                     \
        CV_CPU_DISPATCH(fun, (SCALAR_ARGS_PASS, (const double*)scalar),  \
            CV_CPU_DISPATCH_MODES_ALL);                                  \
    }

DEFINE_SIMD_SAT(addWeighted, add_weighted_loop)
DEFINE_SIMD_S32(addWeighted, add_weighted_loop_d)
DEFINE_SIMD_F32(addWeighted, add_weighted_loop_d)
DEFINE_SIMD_F64(addWeighted, add_weighted_loop_d)


#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_END
#endif

#ifndef SIMD_GUARD
    #define SIMD_GUARD
#endif

}} // cv::hal::