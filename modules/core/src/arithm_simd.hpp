/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_ARITHM_SIMD_HPP__
#define __OPENCV_ARITHM_SIMD_HPP__

namespace cv {

struct NOP {};

#if CV_SSE2 || CV_NEON
#define IF_SIMD(op) op
#else
#define IF_SIMD(op) NOP
#endif


template <typename T>
struct Recip_SIMD
{
    int operator() (const T *, T *, int, double) const
    {
        return 0;
    }
};


#if CV_SIMD128

///////////////////////// RECIPROCAL //////////////////////

template <>
struct Recip_SIMD<uchar>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const uchar * src2, uchar * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_uint16x8 v_zero = v_setzero_u16();

        for ( ; x <= width - 8; x += 8)
        {
            v_uint16x8 v_src2 = v_load_expand(src2 + x);

            v_uint32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(v_reinterpret_as_s32(t0));
            v_float32x4 f1 = v_cvt_f32(v_reinterpret_as_s32(t1));

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_uint16x8 res = v_pack_u(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_pack_store(dst + x, res);
        }

        return x;
    }
};


template <>
struct Recip_SIMD<schar>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const schar * src2, schar * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int16x8 v_zero = v_setzero_s16();

        for ( ; x <= width - 8; x += 8)
        {
            v_int16x8 v_src2 = v_load_expand(src2 + x);

            v_int32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_int16x8 res = v_pack(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_pack_store(dst + x, res);
        }

        return x;
    }
};


template <>
struct Recip_SIMD<ushort>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const ushort * src2, ushort * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_uint16x8 v_zero = v_setzero_u16();

        for ( ; x <= width - 8; x += 8)
        {
            v_uint16x8 v_src2 = v_load(src2 + x);

            v_uint32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(v_reinterpret_as_s32(t0));
            v_float32x4 f1 = v_cvt_f32(v_reinterpret_as_s32(t1));

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_uint16x8 res = v_pack_u(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_store(dst + x, res);
        }

        return x;
    }
};

template <>
struct Recip_SIMD<short>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const short * src2, short * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int16x8 v_zero = v_setzero_s16();

        for ( ; x <= width - 8; x += 8)
        {
            v_int16x8 v_src2 = v_load(src2 + x);

            v_int32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_int16x8 res = v_pack(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_store(dst + x, res);
        }

        return x;
    }
};

template <>
struct Recip_SIMD<int>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const int * src2, int * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int32x4 v_zero = v_setzero_s32();

        for ( ; x <= width - 8; x += 8)
        {
            v_int32x4 t0 = v_load(src2 + x);
            v_int32x4 t1 = v_load(src2 + x + 4);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 res0 = v_round(f0), res1 = v_round(f1);

            res0 = v_select(t0 == v_zero, v_zero, res0);
            res1 = v_select(t1 == v_zero, v_zero, res1);
            v_store(dst + x, res0);
            v_store(dst + x + 4, res1);
        }

        return x;
    }
};


template <>
struct Recip_SIMD<float>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const float * src2, float * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_float32x4 v_zero = v_setzero_f32();

        for ( ; x <= width - 8; x += 8)
        {
            v_float32x4 f0 = v_load(src2 + x);
            v_float32x4 f1 = v_load(src2 + x + 4);

            v_float32x4 res0 = v_scale / f0;
            v_float32x4 res1 = v_scale / f1;

            res0 = v_select(f0 == v_zero, v_zero, res0);
            res1 = v_select(f1 == v_zero, v_zero, res1);

            v_store(dst + x, res0);
            v_store(dst + x + 4, res1);
        }

        return x;
    }
};

#if CV_SIMD128_64F

template <>
struct Recip_SIMD<double>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const double * src2, double * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float64x2 v_scale = v_setall_f64(scale);
        v_float64x2 v_zero = v_setzero_f64();

        for ( ; x <= width - 4; x += 4)
        {
            v_float64x2 f0 = v_load(src2 + x);
            v_float64x2 f1 = v_load(src2 + x + 2);

            v_float64x2 res0 = v_scale / f0;
            v_float64x2 res1 = v_scale / f1;

            res0 = v_select(f0 == v_zero, v_zero, res0);
            res1 = v_select(f1 == v_zero, v_zero, res1);

            v_store(dst + x, res0);
            v_store(dst + x + 2, res1);
        }

        return x;
    }
};

#endif

#endif


}

#endif // __OPENCV_ARITHM_SIMD_HPP__
