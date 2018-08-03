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

// todo: optimize packing, we need a new universal intrinsic
template<template<typename T1, typename Tvec> class OP, typename T1, typename Tvec>
struct cmp_loader_n<sizeof(ushort), OP, T1, Tvec>
{
    typedef OP<T1, Tvec> op;
    enum {step = Tvec::nlanes};

    static inline void l(const T1* src1, const T1* src2, uchar* dst)
    {
        Tvec c0 = op::r(vx_load(src1), vx_load(src2));
        Tvec c1 = op::r(vx_load(src1 + step), vx_load(src2 + step));
        v_store(dst, v_pack(v_reinterpret_as_u16(c0), v_reinterpret_as_u16(c1)));
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
        v_store(dst, v_pack(v_pack(c0, c1), v_pack(c2, c3)));
    }
};

#if CV_SIMD_64F
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

        v_uint16 p0 = v_pack(v_pack(c0, c1), v_pack(c2, c3));
        v_uint16 p1 = v_pack(v_pack(c4, c5), v_pack(c6, c7));
        v_store(dst, v_pack(p0, p1));
    }
};
#endif // CV_SIMD_64F

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
#endif //!CV_SIMD_64F


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

#ifndef ARITHM_DISPATCHING_ONLY
    CV_CPU_OPTIMIZATION_NAMESPACE_END
#endif

#ifndef SIMD_GUARD
    #define SIMD_GUARD
#endif

}} // cv::hal::