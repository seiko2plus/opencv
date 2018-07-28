// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#if ARITHM_USE_IPP

namespace cv { namespace hal {

//=======================================
// Arithmetic and logical operations
// +, -, *, /, &, |, ^, ~, abs ...
//=======================================

#define ARITHM_IPP_BIN(fun, ...)                        \
do {                                                    \
    if (!CV_IPP_CHECK_COND)                             \
        return 0;                                       \
    if (height == 1)                                    \
        step1 = step2 = step = width * sizeof(dst[0]);  \
    if (0 <= CV_INSTRUMENT_FUN_IPP(fun, __VA_ARGS__))   \
    {                                                   \
        CV_IMPL_ADD(CV_IMPL_IPP);                       \
        return 1;                                       \
    }                                                   \
    setIppErrorStatus();                                \
    return 0;                                           \
} while(0)

//=======================================
// Addition
//=======================================

inline int arithm_ipp_add8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                            uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAdd_8u_C1RSfs, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), 0);
}

inline int arithm_ipp_add16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                             ushort* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAdd_16u_C1RSfs, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), 0);
}

inline int arithm_ipp_add16s(const short* src1, size_t step1, const short* src2, size_t step2,
                             short* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAdd_16s_C1RSfs, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height), 0);
}

inline int arithm_ipp_add32f(const float* src1, size_t step1, const float* src2, size_t step2,
                             float* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAdd_32f_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}

#define arithm_ipp_add8s(...)  0
#define arithm_ipp_add32s(...) 0
#define arithm_ipp_add64f(...) 0

//=======================================
// Subtract
//=======================================

inline int arithm_ipp_sub8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                            uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiSub_8u_C1RSfs, src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height), 0);
}

inline int arithm_ipp_sub16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                             ushort* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiSub_16u_C1RSfs, src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height), 0);
}

inline int arithm_ipp_sub16s(const short* src1, size_t step1, const short* src2, size_t step2,
                            short* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiSub_16s_C1RSfs, src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height), 0);
}

inline int arithm_ipp_sub32f(const float* src1, size_t step1, const float* src2, size_t step2,
                            float* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiSub_32f_C1R, src2, (int)step2, src1, (int)step1, dst, (int)step, ippiSize(width, height));
}

#define arithm_ipp_sub8s(...)  0
#define arithm_ipp_sub32s(...) 0
#define arithm_ipp_sub64f(...) 0

///////////////////////////////////////////////////////////////////////////////////////////////////

#define ARITHM_IPP_MIN_MAX(fun, type)                            \
do {                                                             \
    if (!CV_IPP_CHECK_COND)                                      \
        return 0;                                                \
    type* s1 = (type*)src1;                                      \
    type* s2 = (type*)src2;                                      \
    type* d  = dst;                                              \
    if (height == 1)                                             \
        step1 = step2 = step = width * sizeof(dst[0]);           \
    int i = 0;                                                   \
    for(; i < height; i++)                                       \
    {                                                            \
        if (0 > CV_INSTRUMENT_FUN_IPP(fun, s1, s2, d, width))    \
            break;                                               \
        s1 = (type*)((uchar*)s1 + step1);                        \
        s2 = (type*)((uchar*)s2 + step2);                        \
        d  = (type*)((uchar*)d + step);                          \
    }                                                            \
    if (i == height)                                             \
    {                                                            \
        CV_IMPL_ADD(CV_IMPL_IPP);                                \
        return 1;                                                \
    }                                                            \
    setIppErrorStatus();                                         \
    return 0;                                                    \
} while(0)

//=======================================
// Max
//=======================================

inline int arithm_ipp_max8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                           uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMaxEvery_8u, uchar);
}

inline int arithm_ipp_max16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                             ushort* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMaxEvery_16u, ushort);
}

inline int arithm_ipp_max32f(const float* src1, size_t step1, const float* src2, size_t step2,
                             float* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMaxEvery_32f, float);
}

inline int arithm_ipp_max64f(const double* src1, size_t step1, const double* src2, size_t step2,
                             double* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMaxEvery_64f, double);
}

#define arithm_ipp_max8s(...)  0
#define arithm_ipp_max16s(...) 0
#define arithm_ipp_max32s(...) 0

//=======================================
// Min
//=======================================

inline int arithm_ipp_min8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                            uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMinEvery_8u, uchar);
}

inline int arithm_ipp_min16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                            ushort* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMinEvery_16u, ushort);
}

inline int arithm_ipp_min32f(const float* src1, size_t step1, const float* src2,size_t step2,
                             float* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMinEvery_32f, float);
}

inline int arithm_ipp_min64f(const double* src1, size_t step1, const double* src2, size_t step2,
                             double* dst, size_t step, int width, int height)
{
    ARITHM_IPP_MIN_MAX(ippsMinEvery_64f, double);
}

#define arithm_ipp_min8s(...)  0
#define arithm_ipp_min16s(...) 0
#define arithm_ipp_min32s(...) 0

//=======================================
// AbsDiff
//=======================================

inline int arithm_ipp_absdiff8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                                uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAbsDiff_8u_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}

inline int arithm_ipp_absdiff16u(const ushort* src1, size_t step1, const ushort* src2, size_t step2,
                                ushort* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAbsDiff_16u_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}

inline int arithm_ipp_absdiff32f(const float* src1, size_t step1, const float* src2, size_t step2,
                                float* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAbsDiff_32f_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}
#define arithm_ipp_absdiff8s(...)  0
#define arithm_ipp_absdiff16s(...) 0
#define arithm_ipp_absdiff32s(...) 0
#define arithm_ipp_absdiff64f(...) 0

//=======================================
// Logical
//=======================================

inline int arithm_ipp_and8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                            uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiAnd_8u_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}

inline int arithm_ipp_or8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                           uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiOr_8u_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}

inline int arithm_ipp_xor8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
                            uchar* dst, size_t step, int width, int height)
{
    ARITHM_IPP_BIN(ippiXor_8u_C1R, src1, (int)step1, src2, (int)step2, dst, (int)step, ippiSize(width, height));
}

inline int arithm_ipp_not8u(const uchar* src1, size_t step1, uchar* dst, size_t step, int width, int height)
{
    if (!CV_IPP_CHECK_COND)
        return 0;
    if (height == 1)
        step1 = step = width * sizeof(dst[0]);
    if (0 <= CV_INSTRUMENT_FUN_IPP(ippiNot_8u_C1R, src1, (int)step1, dst, (int)step, ippiSize(width, height)))
    {
        CV_IMPL_ADD(CV_IMPL_IPP);
        return 1;
    }
    return 0;
}

/** empty block in case if you have "fun"
#define arithm_ipp_8u(...)  0
#define arithm_ipp_8s(...)  0
#define arithm_ipp_16u(...) 0
#define arithm_ipp_16s(...) 0
#define arithm_ipp_32s(...) 0
#define arithm_ipp_32f(...) 0
#define arithm_ipp_64f(...) 0
**/

}} // cv::hal::

#define ARITHM_CALL_IPP(fun, ...)       \
{                                       \
    if (__CV_EXPAND(fun(__VA_ARGS__)))  \
        return;                         \
}

#endif // ARITHM_USE_IPP


#if !ARITHM_USE_IPP
#define ARITHM_CALL_IPP(...)
#endif