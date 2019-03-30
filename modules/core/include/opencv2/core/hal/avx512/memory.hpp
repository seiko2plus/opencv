// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD512
    #error "Not a standalone header"
#endif

//////////////// Load  ///////////////

template<typename Tp>
inline typename V512_Traits<Tp>::v v512_load(const Tp* ptr)
{ return _mm512_loadu_si512((const __m512i*)ptr); }
template<>
inline v_float32x16 v512_load<float>(const float* ptr)
{ return _mm512_loadu_ps(ptr); }
template<>
inline v_float64x8 v512_load<double>(const double* ptr)
{ return _mm512_loadu_pd(ptr); }

template<typename Tp>
inline typename V512_Traits<Tp>::v v512_load_aligned(const Tp* ptr)
{ return _mm512_load_si512((const __m512i*)ptr); }
template<>
inline v_float32x16 v512_load_aligned<float>(const float* ptr)
{ return _mm512_load_ps(ptr); }
template<>
inline v_float64x8 v512_load_aligned<double>(const double* ptr)
{ return _mm512_load_pd(ptr); }

template<typename Tp>
inline typename V512_Traits<Tp>::v v512_load_low(const Tp* ptr)
{
    __m256i v = _mm256_loadu_si256((const __m256i*)ptr);
    return _mm512_castsi256_si512(v);
}
template<>
inline v_float32x16 v512_load_low<float>(const float* ptr)
{
    __m256 v = _mm256_load_ps(ptr);
    return _mm512_castps256_ps512(v);
}
template<>
inline v_float64x8 v512_load_low<double>(const double* ptr)
{
    __m256d v = _mm256_load_pd(ptr);
    return _mm512_castpd256_pd512(v);
}

template<typename Tp>
inline typename V512_Traits<Tp>::v v512_load_halves(const Tp* ptr0, const Tp* ptr1)
{
    __m256i lo = _mm256_loadu_si256((const __m256i*)ptr0);
    __m256i hi = _mm256_loadu_si256((const __m256i*)ptr1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}
template<>
inline v_float32x16 v512_load_halves<float>(const float* ptr0, const float* ptr1)
{
    __m256 lo = _mm256_load_ps(ptr0);
    __m256 hi = _mm256_load_ps(ptr1);
    return _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1);
}
template<>
inline v_float64x8 v512_load_halves<double>(const double* ptr0, const double* ptr1)
{
    __m256d lo = _mm256_load_pd(ptr0);
    __m256d hi = _mm256_load_pd(ptr1);
    return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1);
}

//////////////// Store ///////////////

template<typename Tp>
inline void v512_store(Tp* ptr, const v_512i& a)
{ _mm512_storeu_si512((__m512i*)ptr, a); }
inline void v512_store(float* ptr, const v_float32x16& a)
{ _mm512_storeu_ps(ptr, a); }
inline void v512_store(double* ptr, const v_float64x8& a)
{ _mm512_storeu_pd(ptr, a); }

template<typename Tp>
inline void v512_store_aligned(Tp* ptr, const v_512i& a)
{ _mm512_store_si512((__m512i*)ptr, a); }
inline void v512_store_aligned(float* ptr, const v_float32x16& a)
{ _mm512_store_ps(ptr, a); }
inline void v512_store_aligned(double* ptr, const v_float64x8& a)
{ _mm512_store_pd(ptr, a); }

template<typename Tp>
inline void v512_store_aligned_nocache(Tp* ptr, const v_512i& a)
{ _mm512_stream_si512((__m512i*)ptr, a); }
inline void v512_store_aligned_nocache(float* ptr, const v_float32x16& a)
{ _mm512_stream_ps(ptr, a); }
inline void v512_store_aligned_nocache(double* ptr, const v_float64x8& a)
{ _mm512_stream_pd(ptr, a); }

template<typename Tp>
inline void v512_store(Tp* ptr, const v_512i& a, hal::StoreMode mode)
{
    switch(mode)
    {
    case hal::STORE_ALIGNED_NOCACHE:
        v512_store_aligned_nocache(ptr, a);
        break;
    case hal::STORE_ALIGNED:
        v512_store_aligned(ptr, a);
        break;
    default:
        v512_store(ptr, a);
    }
}

template<typename Tp>
inline void v512_store_low(Tp* ptr, const v_512i& a)
{ _mm256_storeu_si256((__m256i*)ptr, _mm512_castsi512_si256(a)); }
inline void v512_store_low(float* ptr, const v_float32x16& a)
{ _mm256_storeu_ps(ptr, _mm512_castps512_ps256(a)); }
inline void v512_store_low(double* ptr, const v_float64x8& a)
{ _mm256_storeu_pd(ptr, _mm512_castpd512_pd256(a)); }

template<typename Tp>
inline void v512_store_high(Tp* ptr, const v_512i& a)
{ _mm256_storeu_si256((__m256i*)ptr, _mm512_extracti64x4_epi64(a, 1)); }
inline void v512_store_high(float* ptr, const v_float32x16& a)
{ _mm256_storeu_ps(ptr, _mm512_extractf32x8_ps(a, 1)); }
inline void v512_store_high(double* ptr, const v_float64x8& a)
{ _mm256_storeu_pd(ptr, _mm512_extractf64x4_pd(a, 1)); }

// todo: redirect v512_store to v_store