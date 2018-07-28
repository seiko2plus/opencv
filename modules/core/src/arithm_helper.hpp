// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

//=======================================
// customize options for arithm simd loop
// define it before include this file
//=======================================

// function name
#ifndef SIMD_C_FUN
    #define SIMD_C_FUN simd_loop
#endif

// disable or enable SIMD calls, if CV_SIMD is available
#ifndef SIMD_C_SIMD
    #define SIMD_C_SIMD 1
#endif

// disable or enable, memory align
#ifndef SIMD_C_RALIGN
    #define SIMD_C_RALIGN 0
#endif

// disable or enable, fixed 128-bit
#ifndef SIMD_C_RF128
    #define SIMD_C_RF128 0
#endif

// disable or enable, fixed 64-bit
#ifndef SIMD_C_RF64
    #define SIMD_C_RF64 0
#endif

// disable or enable, unrolling four cpp calls
#ifndef SIMD_C_CPP_UNROLL
    #define SIMD_C_CPP_UNROLL 0
#endif

// disable or enable, unrolling simd128 for an extra call
#ifndef SIMD_C_SIMD128_UNROLL
    #define SIMD_C_SIMD128_UNROLL 0
#endif

// template define
// OP for operation's struct
// T1 for src type
// T2 for dest type
#ifndef SIMD_C_TPL
    #define SIMD_C_TPL class OP, typename T1, typename T2
#endif

// you may need to disable SIMD_C_SIMD128_UNROLL
// if it changed to higher than '1'
#ifndef SIMD_C_LOAD_N
    #define SIMD_C_LOAD_N 1
#endif

// number of sources
#ifndef SIMD_C_SRC_N
    #define SIMD_C_SRC_N 2
#else
    // reupdate to avoid necessary undef from outside
    #undef SIMD_INIT_SRC
#endif

//==========================================
// following macros can be used from outside
//==========================================

#ifndef SIMD_INIT_SRC
    #if SIMD_C_SRC_N == 3
        #define SIMD_INIT_SRC(a, b, c, ...) a, b, c
    #elif SIMD_C_SRC_N == 2
        #define SIMD_INIT_SRC(a, b, ...) a, b
    #else
        #define SIMD_INIT_SRC(a, ...) a
    #endif
#endif

#ifndef SIMD_ARGS
    #define SIMD_ARGS(_t1, _t2)                              \
        SIMD_INIT_SRC(                                       \
            __CV_EXPAND_ARGS(const _t1* src1, size_t step1), \
            __CV_EXPAND_ARGS(const _t1* src2, size_t step2), \
            __CV_EXPAND_ARGS(const _t1* src3, size_t step3), \
        ),                                                   \
        _t2* dst, size_t step, int width, int height
#endif

#ifndef SIMD_ARGS_PASS
    #define SIMD_ARGS_PASS                 \
        SIMD_INIT_SRC(                     \
            __CV_EXPAND_ARGS(src1, step1), \
            __CV_EXPAND_ARGS(src2, step2), \
            __CV_EXPAND_ARGS(src3, step3), \
        ), dst, step, width, height
#endif

//==========================================
// following macros only used internaly
//==========================================

#define __SIMD_OP_PASS_X(_x) SIMD_INIT_SRC(src1 + _x, src2 + _x, src3 + _x), dst + _x

//==========================================
// The loop
//==========================================

#ifdef ARITHM_DEFINITIONS_ONLY

#ifndef __SIMD_GUARD

template<typename T1, typename T2>
static inline bool is_aligned(const T1* src1, const T2* dst)
{ return (((size_t)src1|(size_t)dst) & (CV_SIMD_WIDTH - 1)) == 0; }

template<typename T1, typename T2>
static inline bool is_aligned(const T1* src1, const T1* src2, const T2* dst)
{ return (((size_t)src1|(size_t)src2|(size_t)dst) & (CV_SIMD_WIDTH - 1)) == 0; }

template<typename T1, typename T2>
static inline bool is_aligned(const T1* src1, const T1* src2, const T1* src3, const T2* dst)
{ return (((size_t)src1|(size_t)src2|(size_t)src3|(size_t)dst) & (CV_SIMD_WIDTH - 1)) == 0; }

#endif //__SIMD_GUARD

template<SIMD_C_TPL>
void SIMD_C_FUN(SIMD_ARGS(T1, T2))
{

#if SIMD_C_SIMD && CV_SIMD
    const int wide_step = (CV_SIMD_WIDTH / sizeof(T1)) * SIMD_C_LOAD_N;

    #if SIMD_C_SIMD128_UNROLL && CV_SIMD_WIDTH == 16
    const int wide_step_l = wide_step * 2;
    #else
    const int wide_step_l = wide_step;
    #endif

#endif // SIMD_C_SIMD

    for (; height--;
        SIMD_INIT_SRC(
            src1 = (const T1 *)((const uchar *)src1 + step1),
            src2 = (const T1 *)((const uchar *)src2 + step2),
            src3 = (const T1 *)((const uchar *)src3 + step3)
        ),
        dst  = (T2 *)((uchar *)dst + step)
    )
    {
        int x = 0;

    #if SIMD_C_SIMD && CV_SIMD

        #if SIMD_C_RALIGN
        if (is_aligned(SIMD_INIT_SRC(src1, src2, src3), dst))
        {
            for (; x <= width - wide_step_l; x += wide_step_l)
            {
                OP::run_align(__SIMD_OP_PASS_X(x));
                #if SIMD_C_SIMD128_UNROLL && CV_SIMD_WIDTH == 16
                OP::run_align(__SIMD_OP_PASS_X(x + wide_step));
                #endif
            }
        }
        else
        #endif // SIMD_C_RALIGN

        for (; x <= width - wide_step_l; x += wide_step_l)
        {
            OP::run(__SIMD_OP_PASS_X(x));
            #if SIMD_C_SIMD128_UNROLL && CV_SIMD_WIDTH == 16
            OP::run(__SIMD_OP_PASS_X(x + wide_step));
            #endif
        }

        #if SIMD_C_RF128 && CV_SIMD128 && CV_SIMD_WIDTH > 16
        for (; x <= width - 16/(int)sizeof(T1); x += 16/(int)sizeof(T1))
        { OP::run_f128(__SIMD_OP_PASS_X(x)); }
        #endif

        #if SIMD_C_RF64 && CV_SIMD128
        for (; x <= width - 8/(int)sizeof(T1); x += 8/(int)sizeof(T1))
        { OP::run_f64(__SIMD_OP_PASS_X(x)); }
        #endif

        // v_cleanup();
    #endif // SIMD_C_SIMD

        #if SIMD_C_CPP_UNROLL || CV_ENABLE_UNROLLED
        while (x <= width - 4)
        {
            dst[x] = OP::run(SIMD_INIT_SRC(src1[x], src2[x], src3[x])); x++;
            dst[x] = OP::run(SIMD_INIT_SRC(src1[x], src2[x], src3[x])); x++;
            dst[x] = OP::run(SIMD_INIT_SRC(src1[x], src2[x], src3[x])); x++;
            dst[x] = OP::run(SIMD_INIT_SRC(src1[x], src2[x], src3[x])); x++;
        }
        #endif //SIMD_C_CPP_UNROLL

        for (; x < width; x++)
            dst[x] = OP::run(SIMD_INIT_SRC(src1[x], src2[x], src3[x]));
    }
}

#endif // ARITHM_DEFINITIONS_ONLY


//=========================================
// Declare & Define & Dispatch in one step
//=========================================

#ifdef ARITHM_DECLARATIONS_ONLY
    #undef SIMD_DEF
    #define SIMD_DEF(fun_name, c_type, ...) \
        SIMD_DECLARE_OP(fun_name, c_type)
#endif // ARITHM_DECLARATIONS_ONLY

#ifdef ARITHM_DEFINITIONS_ONLY
    #undef SIMD_DEF
    #define SIMD_DEF(fun_name, c_type, v_type, operation_name, ...)  \
        SIMD_DECLARE_OP(fun_name, c_type)                   \
        SIMD_DEFINE_OP(fun_name, c_type, v_type, operation_name)
#endif // ARITHM_DEFINITIONS_ONLY

#ifdef ARITHM_DISPATCHING_ONLY
    #undef SIMD_DEF
    #define SIMD_DEF(fun_name, c_type, v_type, operation_name, ...)  \
        SIMD_DISPATCH_OP(fun_name, c_type, v_type, operation_name)
#endif // ARITHM_DISPATCHING_ONLY

#ifndef SIMD_GUARD

#define SIMD_DEF_U8(fun, _op) \
    SIMD_DEF(__CV_CAT(fun, 8u), uchar, v_uint8, _op)

#define SIMD_DEF_S8(fun, _op) \
    SIMD_DEF(__CV_CAT(fun, 8s), schar, v_int8,  _op)

#define SIMD_DEF_U16(fun, _op) \
    SIMD_DEF(__CV_CAT(fun, 16u), ushort, v_uint16, _op)

#define SIMD_DEF_S16(fun, _op) \
    SIMD_DEF(__CV_CAT(fun, 16s), short, v_int16,  _op)

#define SIMD_DEF_S32(fun, _op) \
    SIMD_DEF(__CV_CAT(fun, 32s), int, v_int32,  _op)

#define SIMD_DEF_F32(fun, _op) \
    SIMD_DEF(__CV_CAT(fun, 32f), float, v_float32, _op)

#if CV_SIMD_64F
    #define SIMD_DEF_F64(fun, _op) \
        SIMD_DEF(__CV_CAT(fun, 64f), double, v_float64, _op)
#else
    #define SIMD_DEF_F64(fun, _op) \
        SIMD_DEF(__CV_CAT(fun, 64f), double, v_float32 /*dummy*/, _op)
#endif // CV_SIMD_64F

#define SIMD_DEF_SAT(fun, _op) \
    SIMD_DEF_U8(fun, _op)      \
    SIMD_DEF_S8(fun, _op)      \
    SIMD_DEF_U16(fun, _op)     \
    SIMD_DEF_S16(fun, _op)

#define SIMD_DEF_NSAT(fun, _op) \
    SIMD_DEF_S32(fun, _op)      \
    SIMD_DEF_F32(fun, _op)

#endif // SIMD_GUARD

//=======================================
// Cleanup
//=======================================

#undef SIMD_C_FUN
#undef SIMD_C_SIMD
#undef SIMD_C_RALIGN
#undef SIMD_C_RF128
#undef SIMD_C_RF64
#undef SIMD_C_CPP_UNROLL
#undef SIMD_C_SIMD128_UNROLL
#undef SIMD_C_TPL
#undef SIMD_C_LOAD_N
#undef SIMD_C_SRC_N

#undef __SIMD_OP_PASS_X

/*
we need them outside
#undef SIMD_INIT_SRC
#undef SIMD_ARGS
#undef SIMD_ARGS_PASS
*/
/////////////////////////////////////////

#ifndef __SIMD_GUARD
#define __SIMD_GUARD
#endif