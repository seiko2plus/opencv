// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "arithm.simd.hpp"
#include "arithm.simd_declarations.hpp"

namespace cv { namespace hal {

//=======================================

#if (ARITHM_USE_IPP == 1)
static inline void fixSteps(int width, int height, size_t elemSize, size_t& step1, size_t& step2, size_t& step)
{
    if( height == 1 )
        step1 = step2 = step = width*elemSize;
}
#define CALL_IPP_BIN_E_12(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        if (0 <= CV_INSTRUMENT_FUN_IPP(fun, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), 0)) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }

#define CALL_IPP_BIN_E_21(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        if (0 <= CV_INSTRUMENT_FUN_IPP(fun, src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height), 0)) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }

#define CALL_IPP_BIN_12(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        if (0 <= CV_INSTRUMENT_FUN_IPP(fun, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height))) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }

#define CALL_IPP_BIN_21(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        if (0 <= CV_INSTRUMENT_FUN_IPP(fun, src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height))) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }

#else
#define CALL_IPP_BIN_E_12(fun)
#define CALL_IPP_BIN_E_21(fun)
#define CALL_IPP_BIN_12(fun)
#define CALL_IPP_BIN_21(fun)
#endif


//=======================================
// Add
//=======================================

void add8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add8u, cv_hal_add8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_E_12(ippiAdd_8u_C1RSfs)
    CV_CPU_DISPATCH(add8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void add8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add8s, cv_hal_add8s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(add8s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void add16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add16u, cv_hal_add16u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_E_12(ippiAdd_16u_C1RSfs)
    CV_CPU_DISPATCH(add16u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void add16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add16s, cv_hal_add16s, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_E_12(ippiAdd_16s_C1RSfs)
    CV_CPU_DISPATCH(add16s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void add32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add32s, cv_hal_add32s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(add32s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void add32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add32f, cv_hal_add32f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiAdd_32f_C1R)
    CV_CPU_DISPATCH(add32f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void add64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(add64f, cv_hal_add64f, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(add64f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

//=======================================
// Subtract
//=======================================

void sub8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub8u, cv_hal_sub8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_E_21(ippiSub_8u_C1RSfs)
    CV_CPU_DISPATCH(sub8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void sub8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub8s, cv_hal_sub8s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(sub8s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void sub16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub16u, cv_hal_sub16u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_E_21(ippiSub_16u_C1RSfs)
    CV_CPU_DISPATCH(sub16u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void sub16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub16s, cv_hal_sub16s, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_E_21(ippiSub_16s_C1RSfs)
    CV_CPU_DISPATCH(sub16s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void sub32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub32s, cv_hal_sub32s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(sub32s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void sub32f( const float* src1, size_t step1,
                   const float* src2, size_t step2,
                   float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub32f, cv_hal_sub32f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_21(ippiSub_32f_C1R)
    CV_CPU_DISPATCH(sub32f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void sub64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(sub64f, cv_hal_sub64f, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(sub64f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

//=======================================

#if (ARITHM_USE_IPP == 1)
#define CALL_IPP_MIN_MAX(fun, type) \
    CV_IPP_CHECK() \
    { \
        type* s1 = (type*)src1; \
        type* s2 = (type*)src2; \
        type* d  = dst; \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); \
        int i = 0; \
        for(; i < height; i++) \
        { \
            if (0 > CV_INSTRUMENT_FUN_IPP(fun, s1, s2, d, width)) \
                break; \
            s1 = (type*)((uchar*)s1 + step1); \
            s2 = (type*)((uchar*)s2 + step2); \
            d  = (type*)((uchar*)d + step); \
        } \
        if (i == height) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }
#else
#define CALL_IPP_MIN_MAX(fun, type)
#endif

//=======================================
// Max
//=======================================

void max8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max8u, cv_hal_max8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMaxEvery_8u, uchar)
    CV_CPU_DISPATCH(max8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void max8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max8s, cv_hal_max8s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(max8s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void max16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max16u, cv_hal_max16u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMaxEvery_16u, ushort)
    CV_CPU_DISPATCH(max16u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void max16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max16s, cv_hal_max16s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(max16s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void max32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max32s, cv_hal_max32s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(max32s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void max32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max32f, cv_hal_max32f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMaxEvery_32f, float)
    CV_CPU_DISPATCH(max32f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void max64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(max64f, cv_hal_max64f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMaxEvery_64f, double)
    CV_CPU_DISPATCH(max64f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

//=======================================
// Min
//=======================================

void min8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min8u, cv_hal_min8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMinEvery_8u, uchar)
    CV_CPU_DISPATCH(min8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void min8s( const schar* src1, size_t step1,
                   const schar* src2, size_t step2,
                   schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min8s, cv_hal_min8s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(min8s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void min16u( const ushort* src1, size_t step1,
                    const ushort* src2, size_t step2,
                    ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min16u, cv_hal_min16u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMinEvery_16u, ushort)
    CV_CPU_DISPATCH(min16u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void min16s( const short* src1, size_t step1,
                    const short* src2, size_t step2,
                    short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min16s, cv_hal_min16s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(min16s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void min32s( const int* src1, size_t step1,
                    const int* src2, size_t step2,
                    int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min32s, cv_hal_min32s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(min32s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void min32f( const float* src1, size_t step1,
                    const float* src2, size_t step2,
                    float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min32f, cv_hal_min32f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMinEvery_32f, float)
    CV_CPU_DISPATCH(min32f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void min64f( const double* src1, size_t step1,
                    const double* src2, size_t step2,
                    double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(min64f, cv_hal_min64f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_MIN_MAX(ippsMinEvery_64f, double)
    CV_CPU_DISPATCH(min64f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

//=======================================
// AbsDiff
//=======================================

void absdiff8u( const uchar* src1, size_t step1,
                       const uchar* src2, size_t step2,
                       uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff8u, cv_hal_absdiff8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiAbsDiff_8u_C1R)
    CV_CPU_DISPATCH(absdiff8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void absdiff8s( const schar* src1, size_t step1,
                       const schar* src2, size_t step2,
                       schar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff8s, cv_hal_absdiff8s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(absdiff8s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void absdiff16u( const ushort* src1, size_t step1,
                        const ushort* src2, size_t step2,
                        ushort* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff16u, cv_hal_absdiff16u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiAbsDiff_16u_C1R)
    CV_CPU_DISPATCH(absdiff16u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void absdiff16s( const short* src1, size_t step1,
                        const short* src2, size_t step2,
                        short* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff16s, cv_hal_absdiff16s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(absdiff16s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void absdiff32s( const int* src1, size_t step1,
                        const int* src2, size_t step2,
                        int* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff32s, cv_hal_absdiff32s, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(absdiff32s, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void absdiff32f( const float* src1, size_t step1,
                        const float* src2, size_t step2,
                        float* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff32f, cv_hal_absdiff32f, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiAbsDiff_32f_C1R)
    CV_CPU_DISPATCH(absdiff32f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void absdiff64f( const double* src1, size_t step1,
                        const double* src2, size_t step2,
                        double* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(absdiff64f, cv_hal_absdiff64f, src1, step1, src2, step2, dst, step, width, height)
    CV_CPU_DISPATCH(absdiff64f, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

//=======================================
// Logical
//=======================================

#if (ARITHM_USE_IPP == 1)
#define CALL_IPP_UN(fun) \
    CV_IPP_CHECK() \
    { \
        fixSteps(width, height, sizeof(dst[0]), step1, step2, step); (void)src2; \
        if (0 <= CV_INSTRUMENT_FUN_IPP(fun, src1, (int)step1, dst, (int)step, ippiSize(width, height))) \
        { \
            CV_IMPL_ADD(CV_IMPL_IPP); \
            return; \
        } \
        setIppErrorStatus(); \
    }
#else
#define CALL_IPP_UN(fun)
#endif

void and8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(and8u, cv_hal_and8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiAnd_8u_C1R)
    CV_CPU_DISPATCH(and8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void or8u( const uchar* src1, size_t step1,
                  const uchar* src2, size_t step2,
                  uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(or8u, cv_hal_or8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiOr_8u_C1R)
    CV_CPU_DISPATCH(or8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void xor8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(xor8u, cv_hal_xor8u, src1, step1, src2, step2, dst, step, width, height)
    CALL_IPP_BIN_12(ippiXor_8u_C1R)
    CV_CPU_DISPATCH(xor8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

void not8u( const uchar* src1, size_t step1,
                   const uchar* src2, size_t step2,
                   uchar* dst, size_t step, int width, int height, void* )
{
    CALL_HAL(not8u, cv_hal_not8u, src1, step1, dst, step, width, height)
    CALL_IPP_UN(ippiNot_8u_C1R)
    CV_CPU_DISPATCH(not8u, (src1, step1, src2, step2, dst, step, width, height), CV_CPU_DISPATCH_MODES_ALL);
}

}} // cv::hal::