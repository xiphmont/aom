#include <smmintrin.h>

#include "./aom_config.h"
#include "./av1_rtcd.h"
#include "av1/common/restoration.h"

/* Calculate four consecutive entries of the intermediate A and B arrays
   (corresponding to the first loop in the C version of
   av1_selfguided_restoration)
*/
static void calc_block(__m128i sum, __m128i sum_sq, __m128i n,
                       __m128i one_over_n, __m128i s, int bit_depth, int idx,
                       int32_t *A, int32_t *B) {
  __m128i a, b;
#if CONFIG_AOM_HIGHBITDEPTH
  __m128i rounding_a = _mm_set1_epi32((1 << (2 * (bit_depth - 8))) >> 1);
  __m128i rounding_b = _mm_set1_epi32((1 << (bit_depth - 8)) >> 1);
  a = _mm_srl_epi32(_mm_add_epi32(sum_sq, rounding_a),
                    _mm_set1_epi32(2 * (bit_depth - 8)));
  b = _mm_srl_epi32(_mm_add_epi32(sum, rounding_b),
                    _mm_set1_epi32(bit_depth - 8));
  a = _mm_mullo_epi32(a, n);
  b = _mm_mullo_epi32(b, b);
  __m128i p = _mm_sub_epi32(_mm_max_epi32(a, b), b);
#else
  (void)bit_depth;
  a = _mm_mullo_epi32(sum_sq, n);
  b = _mm_mullo_epi32(sum, sum);
  __m128i p = _mm_sub_epi32(a, b);
#endif

  __m128i rounding_z = _mm_set1_epi32((1 << SGRPROJ_MTABLE_BITS) >> 1);
  __m128i z = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(p, s), rounding_z),
                             SGRPROJ_MTABLE_BITS);
  z = _mm_min_epi32(z, _mm_set1_epi32(255));

  // 'Gather' type instructions are not available pre-AVX2, so synthesize a
  // gather using scalar loads.
  __m128i a_res = _mm_set_epi32(x_by_xplus1[_mm_extract_epi32(z, 3)],
                                x_by_xplus1[_mm_extract_epi32(z, 2)],
                                x_by_xplus1[_mm_extract_epi32(z, 1)],
                                x_by_xplus1[_mm_extract_epi32(z, 0)]);

  _mm_store_si128((__m128i *)&A[idx], a_res);

  __m128i rounding_res = _mm_set1_epi32((1 << SGRPROJ_RECIP_BITS) >> 1);
  __m128i a_complement = _mm_sub_epi32(_mm_set1_epi32(SGRPROJ_SGR), a_res);
  __m128i b_int =
      _mm_mullo_epi32(a_complement, _mm_mullo_epi32(sum, one_over_n));
  __m128i b_res =
      _mm_srli_epi32(_mm_add_epi32(b_int, rounding_res), SGRPROJ_RECIP_BITS);

  _mm_store_si128((__m128i *)&B[idx], b_res);
}

static void selfguided_restoration_1(uint8_t *src, int width, int height,
                                     int src_stride, int eps, int bit_depth,
                                     int32_t *A, int32_t *B, int buf_stride) {
  int i, j;

  // Vertical sum
  assert(!(width & 3));
  for (j = 0; j < width; j += 4) {
    __m128i a, b, x, y, x2, y2;
    __m128i sum, sum_sq, tmp;

    a = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[j]));
    b = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[src_stride + j]));

    sum = _mm_cvtepi16_epi32(_mm_add_epi16(a, b));
    tmp = _mm_unpacklo_epi16(a, b);
    sum_sq = _mm_madd_epi16(tmp, tmp);

    _mm_store_si128((__m128i *)&B[j], sum);
    _mm_store_si128((__m128i *)&A[j], sum_sq);

    x = _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&src[2 * src_stride + j]));
    sum = _mm_add_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_add_epi32(sum_sq, x2);

    for (i = 1; i < height - 2; ++i) {
      _mm_store_si128((__m128i *)&B[i * buf_stride + j], sum);
      _mm_store_si128((__m128i *)&A[i * buf_stride + j], sum_sq);

      x = _mm_cvtepu8_epi32(
          _mm_loadl_epi64((__m128i *)&src[(i - 1) * src_stride + j]));
      y = _mm_cvtepu8_epi32(
          _mm_loadl_epi64((__m128i *)&src[(i + 2) * src_stride + j]));

      sum = _mm_add_epi32(sum, _mm_sub_epi32(y, x));

      x2 = _mm_mullo_epi32(x, x);
      y2 = _mm_mullo_epi32(y, y);

      sum_sq = _mm_add_epi32(sum_sq, _mm_sub_epi32(y2, x2));
    }
    _mm_store_si128((__m128i *)&B[i * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[i * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(
        _mm_loadl_epi64((__m128i *)&src[(i - 1) * src_stride + j]));
    sum = _mm_sub_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_sub_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[(i + 1) * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[(i + 1) * buf_stride + j], sum_sq);
  }

  // Horizontal sum
  for (i = 0; i < height; ++i) {
    int h = AOMMIN(2, height - i) + AOMMIN(1, i);

    __m128i a1 = _mm_loadu_si128((__m128i *)&A[i * buf_stride]);
    __m128i b1 = _mm_loadu_si128((__m128i *)&B[i * buf_stride]);
    __m128i a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + 4]);
    __m128i b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + 4]);

    // Note: The _mm_slli_si128 call sets up a register containing
    // {0, A[i * buf_stride], ..., A[i * buf_stride + 2]},
    // so that the first element of 'sum' (which should only add two values
    // together) ends up calculated correctly.
    __m128i sum_ = _mm_add_epi32(_mm_slli_si128(b1, 4),
                                 _mm_add_epi32(b1, _mm_alignr_epi8(b2, b1, 4)));
    __m128i sum_sq_ = _mm_add_epi32(
        _mm_slli_si128(a1, 4), _mm_add_epi32(a1, _mm_alignr_epi8(a2, a1, 4)));
    __m128i n = _mm_set_epi32(3 * h, 3 * h, 3 * h, 2 * h);
    __m128i one_over_n =
        _mm_set_epi32(one_by_x[3 * h - 1], one_by_x[3 * h - 1],
                      one_by_x[3 * h - 1], one_by_x[2 * h - 1]);
    __m128i s = _mm_set_epi32(
        sgrproj_mtable[eps - 1][3 * h - 1], sgrproj_mtable[eps - 1][3 * h - 1],
        sgrproj_mtable[eps - 1][3 * h - 1], sgrproj_mtable[eps - 1][2 * h - 1]);
    calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride, A,
               B);

    n = _mm_set1_epi32(3 * h);
    one_over_n = _mm_set1_epi32(one_by_x[3 * h - 1]);
    s = _mm_set1_epi32(sgrproj_mtable[eps - 1][3 * h - 1]);

    // Re-align a1 and b1 so that they start at index i * buf_stride + 3
    a1 = _mm_alignr_epi8(a2, a1, 12);
    b1 = _mm_alignr_epi8(b2, b1, 12);
    a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + 7]);
    b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + 7]);

    for (j = 4; j < width - 4; j += 4) {
      /* Loop invariant: At this point,
         a1 = original A[i * buf_stride + j - 1 : i * buf_stride + j + 3]
         a2 = original A[i * buf_stride + j + 3 : i * buf_stride + j + 7]
         and similar for b1,b2 and B
      */
      sum_ = _mm_add_epi32(b1, _mm_add_epi32(_mm_alignr_epi8(b2, b1, 4),
                                             _mm_alignr_epi8(b2, b1, 8)));
      sum_sq_ = _mm_add_epi32(a1, _mm_add_epi32(_mm_alignr_epi8(a2, a1, 4),
                                                _mm_alignr_epi8(a2, a1, 8)));
      calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride + j,
                 A, B);

      a1 = a2;
      a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + j + 7]);
      b1 = b2;
      b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + j + 7]);
    }
    // Zero out the data loaded from "off the edge" of the array
    __m128i zero = _mm_setzero_si128();
    a2 = _mm_blend_epi16(a2, zero, 0xfc);
    b2 = _mm_blend_epi16(b2, zero, 0xfc);

    sum_ = _mm_add_epi32(b1, _mm_add_epi32(_mm_alignr_epi8(b2, b1, 4),
                                           _mm_alignr_epi8(b2, b1, 8)));
    sum_sq_ = _mm_add_epi32(a1, _mm_add_epi32(_mm_alignr_epi8(a2, a1, 4),
                                              _mm_alignr_epi8(a2, a1, 8)));
    n = _mm_set_epi32(2 * h, 3 * h, 3 * h, 3 * h);
    one_over_n = _mm_set_epi32(one_by_x[2 * h - 1], one_by_x[3 * h - 1],
                               one_by_x[3 * h - 1], one_by_x[3 * h - 1]);
    s = _mm_set_epi32(
        sgrproj_mtable[eps - 1][2 * h - 1], sgrproj_mtable[eps - 1][3 * h - 1],
        sgrproj_mtable[eps - 1][3 * h - 1], sgrproj_mtable[eps - 1][3 * h - 1]);
    calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride + j,
               A, B);
  }
}

static void selfguided_restoration_2(uint8_t *src, int width, int height,
                                     int src_stride, int eps, int bit_depth,
                                     int32_t *A, int32_t *B, int buf_stride) {
  int i, j;

  // Vertical sum
  assert(!(width & 3));
  for (j = 0; j < width; j += 4) {
    __m128i a, b, c, c2, x, y, x2, y2;
    __m128i sum, sum_sq, tmp;

    a = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[j]));
    b = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[src_stride + j]));
    c = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[2 * src_stride + j]));

    sum = _mm_cvtepi16_epi32(_mm_add_epi16(_mm_add_epi16(a, b), c));
    // Important: Since c may be up to 2^8, the result on squaring may
    // be up to 2^16. So we need to zero-extend, not sign-extend.
    c2 = _mm_cvtepu16_epi32(_mm_mullo_epi16(c, c));
    tmp = _mm_unpacklo_epi16(a, b);
    sum_sq = _mm_add_epi32(_mm_madd_epi16(tmp, tmp), c2);

    _mm_store_si128((__m128i *)&B[j], sum);
    _mm_store_si128((__m128i *)&A[j], sum_sq);

    x = _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&src[3 * src_stride + j]));
    sum = _mm_add_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_add_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&src[4 * src_stride + j]));
    sum = _mm_add_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_add_epi32(sum_sq, x2);

    for (i = 2; i < height - 3; ++i) {
      _mm_store_si128((__m128i *)&B[i * buf_stride + j], sum);
      _mm_store_si128((__m128i *)&A[i * buf_stride + j], sum_sq);

      x = _mm_cvtepu8_epi32(
          _mm_loadl_epi64((__m128i *)&src[(i - 2) * src_stride + j]));
      y = _mm_cvtepu8_epi32(
          _mm_loadl_epi64((__m128i *)&src[(i + 3) * src_stride + j]));

      sum = _mm_add_epi32(sum, _mm_sub_epi32(y, x));

      x2 = _mm_mullo_epi32(x, x);
      y2 = _mm_mullo_epi32(y, y);

      sum_sq = _mm_add_epi32(sum_sq, _mm_sub_epi32(y2, x2));
    }
    _mm_store_si128((__m128i *)&B[i * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[i * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(
        _mm_loadl_epi64((__m128i *)&src[(i - 2) * src_stride + j]));
    sum = _mm_sub_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_sub_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[(i + 1) * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[(i + 1) * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(
        _mm_loadl_epi64((__m128i *)&src[(i - 1) * src_stride + j]));
    sum = _mm_sub_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_sub_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[(i + 2) * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[(i + 2) * buf_stride + j], sum_sq);
  }

  // Horizontal sum
  for (i = 0; i < height; ++i) {
    int h = AOMMIN(3, height - i) + AOMMIN(2, i);

    __m128i a1 = _mm_loadu_si128((__m128i *)&A[i * buf_stride]);
    __m128i b1 = _mm_loadu_si128((__m128i *)&B[i * buf_stride]);
    __m128i a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + 4]);
    __m128i b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + 4]);

    __m128i sum_ = _mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(b1, 8), _mm_slli_si128(b1, 4)),
            _mm_add_epi32(b1, _mm_alignr_epi8(b2, b1, 4))),
        _mm_alignr_epi8(b2, b1, 8));
    __m128i sum_sq_ = _mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(a1, 8), _mm_slli_si128(a1, 4)),
            _mm_add_epi32(a1, _mm_alignr_epi8(a2, a1, 4))),
        _mm_alignr_epi8(a2, a1, 8));

    __m128i n = _mm_set_epi32(5 * h, 5 * h, 4 * h, 3 * h);
    __m128i one_over_n =
        _mm_set_epi32(one_by_x[5 * h - 1], one_by_x[5 * h - 1],
                      one_by_x[4 * h - 1], one_by_x[3 * h - 1]);
    __m128i s = _mm_set_epi32(
        sgrproj_mtable[eps - 1][5 * h - 1], sgrproj_mtable[eps - 1][5 * h - 1],
        sgrproj_mtable[eps - 1][4 * h - 1], sgrproj_mtable[eps - 1][3 * h - 1]);
    calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride, A,
               B);

    // Re-align a1 and b1 so that they start at index i * buf_stride + 2
    a1 = _mm_alignr_epi8(a2, a1, 8);
    b1 = _mm_alignr_epi8(b2, b1, 8);
    a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + 6]);
    b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + 6]);

    n = _mm_set1_epi32(5 * h);
    one_over_n = _mm_set1_epi32(one_by_x[5 * h - 1]);
    s = _mm_set1_epi32(sgrproj_mtable[eps - 1][5 * h - 1]);

    for (j = 4; j < width - 4; j += 4) {
      /* Loop invariant: At this point,
         a1 = original A[i * buf_stride + j - 2 : i * buf_stride + j + 2]
         a2 = original A[i * buf_stride + j + 2 : i * buf_stride + j + 6]
         and similar for b1,b2 and B
      */
      sum_ = _mm_add_epi32(
          _mm_add_epi32(b1, _mm_add_epi32(_mm_alignr_epi8(b2, b1, 4),
                                          _mm_alignr_epi8(b2, b1, 8))),
          _mm_add_epi32(_mm_alignr_epi8(b2, b1, 12), b2));
      sum_sq_ = _mm_add_epi32(
          _mm_add_epi32(a1, _mm_add_epi32(_mm_alignr_epi8(a2, a1, 4),
                                          _mm_alignr_epi8(a2, a1, 8))),
          _mm_add_epi32(_mm_alignr_epi8(a2, a1, 12), a2));

      calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride + j,
                 A, B);

      a1 = a2;
      a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + j + 6]);
      b1 = b2;
      b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + j + 6]);
    }
    // Zero out the data loaded from "off the edge" of the array
    __m128i zero = _mm_setzero_si128();
    a2 = _mm_blend_epi16(a2, zero, 0xf0);
    b2 = _mm_blend_epi16(b2, zero, 0xf0);

    sum_ = _mm_add_epi32(
        _mm_add_epi32(b1, _mm_add_epi32(_mm_alignr_epi8(b2, b1, 4),
                                        _mm_alignr_epi8(b2, b1, 8))),
        _mm_add_epi32(_mm_alignr_epi8(b2, b1, 12), b2));
    sum_sq_ = _mm_add_epi32(
        _mm_add_epi32(a1, _mm_add_epi32(_mm_alignr_epi8(a2, a1, 4),
                                        _mm_alignr_epi8(a2, a1, 8))),
        _mm_add_epi32(_mm_alignr_epi8(a2, a1, 12), a2));

    n = _mm_set_epi32(3 * h, 4 * h, 5 * h, 5 * h);
    one_over_n = _mm_set_epi32(one_by_x[3 * h - 1], one_by_x[4 * h - 1],
                               one_by_x[5 * h - 1], one_by_x[5 * h - 1]);
    s = _mm_set_epi32(
        sgrproj_mtable[eps - 1][3 * h - 1], sgrproj_mtable[eps - 1][4 * h - 1],
        sgrproj_mtable[eps - 1][5 * h - 1], sgrproj_mtable[eps - 1][5 * h - 1]);
    calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride + j,
               A, B);
  }
}

static void selfguided_restoration_3(uint8_t *src, int width, int height,
                                     int src_stride, int eps, int bit_depth,
                                     int32_t *A, int32_t *B, int buf_stride) {
  int i, j;

  // Vertical sum over 7-pixel regions, 4 columns at a time
  assert(!(width & 3));
  for (j = 0; j < width; j += 4) {
    __m128i a, b, c, d, x, y, x2, y2;
    __m128i sum, sum_sq, tmp, tmp2;

    a = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[j]));
    b = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[src_stride + j]));
    c = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[2 * src_stride + j]));
    d = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&src[3 * src_stride + j]));

    sum = _mm_cvtepi16_epi32(
        _mm_add_epi16(_mm_add_epi16(a, b), _mm_add_epi16(c, d)));
    tmp = _mm_unpacklo_epi16(a, b);
    tmp2 = _mm_unpacklo_epi16(c, d);
    sum_sq =
        _mm_add_epi32(_mm_madd_epi16(tmp, tmp), _mm_madd_epi16(tmp2, tmp2));

    _mm_store_si128((__m128i *)&B[j], sum);
    _mm_store_si128((__m128i *)&A[j], sum_sq);

    x = _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&src[4 * src_stride + j]));
    sum = _mm_add_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_add_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&src[5 * src_stride + j]));
    sum = _mm_add_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_add_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[2 * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[2 * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(_mm_loadl_epi64((__m128i *)&src[6 * src_stride + j]));
    sum = _mm_add_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_add_epi32(sum_sq, x2);

    for (i = 3; i < height - 4; ++i) {
      _mm_store_si128((__m128i *)&B[i * buf_stride + j], sum);
      _mm_store_si128((__m128i *)&A[i * buf_stride + j], sum_sq);

      x = _mm_cvtepu8_epi32(
          _mm_loadl_epi64((__m128i *)&src[(i - 3) * src_stride + j]));
      y = _mm_cvtepu8_epi32(
          _mm_loadl_epi64((__m128i *)&src[(i + 4) * src_stride + j]));

      sum = _mm_add_epi32(sum, _mm_sub_epi32(y, x));

      x2 = _mm_mullo_epi32(x, x);
      y2 = _mm_mullo_epi32(y, y);

      sum_sq = _mm_add_epi32(sum_sq, _mm_sub_epi32(y2, x2));
    }
    _mm_store_si128((__m128i *)&B[i * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[i * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(
        _mm_loadl_epi64((__m128i *)&src[(i - 3) * src_stride + j]));
    sum = _mm_sub_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_sub_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[(i + 1) * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[(i + 1) * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(
        _mm_loadl_epi64((__m128i *)&src[(i - 2) * src_stride + j]));
    sum = _mm_sub_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_sub_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[(i + 2) * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[(i + 2) * buf_stride + j], sum_sq);

    x = _mm_cvtepu8_epi32(
        _mm_loadl_epi64((__m128i *)&src[(i - 1) * src_stride + j]));
    sum = _mm_sub_epi32(sum, x);
    x2 = _mm_mullo_epi32(x, x);
    sum_sq = _mm_sub_epi32(sum_sq, x2);

    _mm_store_si128((__m128i *)&B[(i + 3) * buf_stride + j], sum);
    _mm_store_si128((__m128i *)&A[(i + 3) * buf_stride + j], sum_sq);
  }

  // Horizontal sum over 7-pixel regions of dst
  for (i = 0; i < height; ++i) {
    int h = AOMMIN(4, height - i) + AOMMIN(3, i);

    __m128i a1 = _mm_loadu_si128((__m128i *)&A[i * buf_stride]);
    __m128i b1 = _mm_loadu_si128((__m128i *)&B[i * buf_stride]);
    __m128i a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + 4]);
    __m128i b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + 4]);

    __m128i sum_ = _mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(b1, 12), _mm_slli_si128(b1, 8)),
            _mm_add_epi32(_mm_slli_si128(b1, 4), b1)),
        _mm_add_epi32(_mm_add_epi32(_mm_alignr_epi8(b2, b1, 4),
                                    _mm_alignr_epi8(b2, b1, 8)),
                      _mm_alignr_epi8(b2, b1, 12)));
    __m128i sum_sq_ = _mm_add_epi32(
        _mm_add_epi32(
            _mm_add_epi32(_mm_slli_si128(a1, 12), _mm_slli_si128(a1, 8)),
            _mm_add_epi32(_mm_slli_si128(a1, 4), a1)),
        _mm_add_epi32(_mm_add_epi32(_mm_alignr_epi8(a2, a1, 4),
                                    _mm_alignr_epi8(a2, a1, 8)),
                      _mm_alignr_epi8(a2, a1, 12)));

    __m128i n = _mm_set_epi32(7 * h, 6 * h, 5 * h, 4 * h);
    __m128i one_over_n =
        _mm_set_epi32(one_by_x[7 * h - 1], one_by_x[6 * h - 1],
                      one_by_x[5 * h - 1], one_by_x[4 * h - 1]);
    __m128i s = _mm_set_epi32(
        sgrproj_mtable[eps - 1][7 * h - 1], sgrproj_mtable[eps - 1][6 * h - 1],
        sgrproj_mtable[eps - 1][5 * h - 1], sgrproj_mtable[eps - 1][4 * h - 1]);
    calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride, A,
               B);

    // Re-align a1 and b1 so that they start at index i * buf_stride + 1
    a1 = _mm_alignr_epi8(a2, a1, 4);
    b1 = _mm_alignr_epi8(b2, b1, 4);
    a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + 5]);
    b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + 5]);

    n = _mm_set1_epi32(7 * h);
    one_over_n = _mm_set1_epi32(one_by_x[7 * h - 1]);
    s = _mm_set1_epi32(sgrproj_mtable[eps - 1][7 * h - 1]);

    for (j = 4; j < width - 4; j += 4) {
      __m128i a3 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + j + 5]);
      __m128i b3 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + j + 5]);
      /* Loop invariant: At this point,
         a1 = original A[i * buf_stride + j - 3 : i * buf_stride + j + 1]
         a2 = original A[i * buf_stride + j + 1 : i * buf_stride + j + 5]
         a3 = original A[i * buf_stride + j + 5 : i * buf_stride + j + 9]
         and similar for b1,b2,b3 and B
      */
      sum_ = _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(b1, _mm_alignr_epi8(b2, b1, 4)),
                        _mm_add_epi32(_mm_alignr_epi8(b2, b1, 8),
                                      _mm_alignr_epi8(b2, b1, 12))),
          _mm_add_epi32(_mm_add_epi32(b2, _mm_alignr_epi8(b3, b2, 4)),
                        _mm_alignr_epi8(b3, b2, 8)));
      sum_sq_ = _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(a1, _mm_alignr_epi8(a2, a1, 4)),
                        _mm_add_epi32(_mm_alignr_epi8(a2, a1, 8),
                                      _mm_alignr_epi8(a2, a1, 12))),
          _mm_add_epi32(_mm_add_epi32(a2, _mm_alignr_epi8(a3, a2, 4)),
                        _mm_alignr_epi8(a3, a2, 8)));

      calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride + j,
                 A, B);

      a1 = a2;
      a2 = _mm_loadu_si128((__m128i *)&A[i * buf_stride + j + 5]);
      b1 = b2;
      b2 = _mm_loadu_si128((__m128i *)&B[i * buf_stride + j + 5]);
    }
    // Zero out the data loaded from "off the edge" of the array
    __m128i zero = _mm_setzero_si128();
    a2 = _mm_blend_epi16(a2, zero, 0xc0);
    b2 = _mm_blend_epi16(b2, zero, 0xc0);

    sum_ = _mm_add_epi32(
        _mm_add_epi32(_mm_add_epi32(b1, _mm_alignr_epi8(b2, b1, 4)),
                      _mm_add_epi32(_mm_alignr_epi8(b2, b1, 8),
                                    _mm_alignr_epi8(b2, b1, 12))),
        _mm_add_epi32(_mm_add_epi32(b2, _mm_alignr_epi8(zero, b2, 4)),
                      _mm_alignr_epi8(zero, b2, 8)));
    sum_sq_ = _mm_add_epi32(
        _mm_add_epi32(_mm_add_epi32(a1, _mm_alignr_epi8(a2, a1, 4)),
                      _mm_add_epi32(_mm_alignr_epi8(a2, a1, 8),
                                    _mm_alignr_epi8(a2, a1, 12))),
        _mm_add_epi32(_mm_add_epi32(a2, _mm_alignr_epi8(zero, a2, 4)),
                      _mm_alignr_epi8(zero, a2, 8)));

    n = _mm_set_epi32(4 * h, 5 * h, 6 * h, 7 * h);
    one_over_n = _mm_set_epi32(one_by_x[4 * h - 1], one_by_x[5 * h - 1],
                               one_by_x[6 * h - 1], one_by_x[7 * h - 1]);
    s = _mm_set_epi32(
        sgrproj_mtable[eps - 1][4 * h - 1], sgrproj_mtable[eps - 1][5 * h - 1],
        sgrproj_mtable[eps - 1][6 * h - 1], sgrproj_mtable[eps - 1][7 * h - 1]);
    calc_block(sum_, sum_sq_, n, one_over_n, s, bit_depth, i * buf_stride + j,
               A, B);
  }
}

void av1_selfguided_restoration_sse4_1(uint8_t *dgd, int width, int height,
                                       int stride, int32_t *dst, int dst_stride,
                                       int bit_depth, int r, int eps,
                                       int32_t *tmpbuf) {
  int32_t *A = tmpbuf;
  int32_t *B = A + SGRPROJ_OUTBUF_SIZE;
  int i, j;
  // Adjusting the stride of A and B here appears to avoid bad cache effects,
  // leading to a significant speed improvement.
  // We also align the stride to a multiple of 16 bytes for efficiency.
  int buf_stride = ((width + 3) & ~3) + 16;

  // Don't filter tiles with dimensions < 5 on any axis
  if ((width < 5) || (height < 5)) return;

  if (r == 1) {
    selfguided_restoration_1(dgd, width, height, stride, eps, bit_depth, A, B,
                             buf_stride);
  } else if (r == 2) {
    selfguided_restoration_2(dgd, width, height, stride, eps, bit_depth, A, B,
                             buf_stride);
  } else if (r == 3) {
    selfguided_restoration_3(dgd, width, height, stride, eps, bit_depth, A, B,
                             buf_stride);
  } else {
    assert(0);
  }

  {
    i = 0;
    j = 0;
    {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = 3 * A[k] + 2 * A[k + 1] + 2 * A[k + buf_stride] +
                        A[k + buf_stride + 1];
      const int32_t b = 3 * B[k] + 2 * B[k + 1] + 2 * B[k + buf_stride] +
                        B[k + buf_stride + 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
    for (j = 1; j < width - 1; ++j) {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = A[k] + 2 * (A[k - 1] + A[k + 1]) + A[k + buf_stride] +
                        A[k + buf_stride - 1] + A[k + buf_stride + 1];
      const int32_t b = B[k] + 2 * (B[k - 1] + B[k + 1]) + B[k + buf_stride] +
                        B[k + buf_stride - 1] + B[k + buf_stride + 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
    j = width - 1;
    {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = 3 * A[k] + 2 * A[k - 1] + 2 * A[k + buf_stride] +
                        A[k + buf_stride - 1];
      const int32_t b = 3 * B[k] + 2 * B[k - 1] + 2 * B[k + buf_stride] +
                        B[k + buf_stride - 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
  }
  for (i = 1; i < height - 1; ++i) {
    j = 0;
    {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = A[k] + 2 * (A[k - buf_stride] + A[k + buf_stride]) +
                        A[k + 1] + A[k - buf_stride + 1] +
                        A[k + buf_stride + 1];
      const int32_t b = B[k] + 2 * (B[k - buf_stride] + B[k + buf_stride]) +
                        B[k + 1] + B[k - buf_stride + 1] +
                        B[k + buf_stride + 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }

    // Vectorize the innermost loop
    for (j = 1; j < width - 1; j += 4) {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 5;

      __m128i tmp0 = _mm_loadu_si128((__m128i *)&A[k - 1 - buf_stride]);
      __m128i tmp1 = _mm_loadu_si128((__m128i *)&A[k + 3 - buf_stride]);
      __m128i tmp2 = _mm_loadu_si128((__m128i *)&A[k - 1]);
      __m128i tmp3 = _mm_loadu_si128((__m128i *)&A[k + 3]);
      __m128i tmp4 = _mm_loadu_si128((__m128i *)&A[k - 1 + buf_stride]);
      __m128i tmp5 = _mm_loadu_si128((__m128i *)&A[k + 3 + buf_stride]);

      __m128i a0 = _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(_mm_alignr_epi8(tmp3, tmp2, 4), tmp2),
                        _mm_add_epi32(_mm_alignr_epi8(tmp3, tmp2, 8),
                                      _mm_alignr_epi8(tmp5, tmp4, 4))),
          _mm_alignr_epi8(tmp1, tmp0, 4));
      __m128i a1 = _mm_add_epi32(_mm_add_epi32(tmp0, tmp4),
                                 _mm_add_epi32(_mm_alignr_epi8(tmp1, tmp0, 8),
                                               _mm_alignr_epi8(tmp5, tmp4, 8)));
      __m128i a = _mm_sub_epi32(_mm_slli_epi32(_mm_add_epi32(a0, a1), 2), a1);

      __m128i tmp6 = _mm_loadu_si128((__m128i *)&B[k - 1 - buf_stride]);
      __m128i tmp7 = _mm_loadu_si128((__m128i *)&B[k + 3 - buf_stride]);
      __m128i tmp8 = _mm_loadu_si128((__m128i *)&B[k - 1]);
      __m128i tmp9 = _mm_loadu_si128((__m128i *)&B[k + 3]);
      __m128i tmp10 = _mm_loadu_si128((__m128i *)&B[k - 1 + buf_stride]);
      __m128i tmp11 = _mm_loadu_si128((__m128i *)&B[k + 3 + buf_stride]);

      __m128i b0 = _mm_add_epi32(
          _mm_add_epi32(_mm_add_epi32(_mm_alignr_epi8(tmp9, tmp8, 4), tmp8),
                        _mm_add_epi32(_mm_alignr_epi8(tmp9, tmp8, 8),
                                      _mm_alignr_epi8(tmp11, tmp10, 4))),
          _mm_alignr_epi8(tmp7, tmp6, 4));
      __m128i b1 =
          _mm_add_epi32(_mm_add_epi32(tmp6, tmp10),
                        _mm_add_epi32(_mm_alignr_epi8(tmp7, tmp6, 8),
                                      _mm_alignr_epi8(tmp11, tmp10, 8)));
      __m128i b = _mm_sub_epi32(_mm_slli_epi32(_mm_add_epi32(b0, b1), 2), b1);

      __m128i src = _mm_cvtepu8_epi32(_mm_loadu_si128((__m128i *)&dgd[l]));

      __m128i rounding = _mm_set1_epi32(
          (1 << (SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS)) >> 1);
      __m128i v = _mm_add_epi32(_mm_mullo_epi32(a, src), b);
      __m128i w = _mm_srai_epi32(_mm_add_epi32(v, rounding),
                                 SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
      _mm_storeu_si128((__m128i *)&dst[m], w);
    }

    // Deal with any extra pixels at the right-hand edge of the frame
    // (typically have 2 such pixels, but may have anywhere between 0 and 3)
    for (; j < width - 1; ++j) {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 5;
      const int32_t a =
          (A[k] + A[k - 1] + A[k + 1] + A[k - buf_stride] + A[k + buf_stride]) *
              4 +
          (A[k - 1 - buf_stride] + A[k - 1 + buf_stride] +
           A[k + 1 - buf_stride] + A[k + 1 + buf_stride]) *
              3;
      const int32_t b =
          (B[k] + B[k - 1] + B[k + 1] + B[k - buf_stride] + B[k + buf_stride]) *
              4 +
          (B[k - 1 - buf_stride] + B[k - 1 + buf_stride] +
           B[k + 1 - buf_stride] + B[k + 1 + buf_stride]) *
              3;
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }

    j = width - 1;
    {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = A[k] + 2 * (A[k - buf_stride] + A[k + buf_stride]) +
                        A[k - 1] + A[k - buf_stride - 1] +
                        A[k + buf_stride - 1];
      const int32_t b = B[k] + 2 * (B[k - buf_stride] + B[k + buf_stride]) +
                        B[k - 1] + B[k - buf_stride - 1] +
                        B[k + buf_stride - 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
  }

  {
    i = height - 1;
    j = 0;
    {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = 3 * A[k] + 2 * A[k + 1] + 2 * A[k - buf_stride] +
                        A[k - buf_stride + 1];
      const int32_t b = 3 * B[k] + 2 * B[k + 1] + 2 * B[k - buf_stride] +
                        B[k - buf_stride + 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
    for (j = 1; j < width - 1; ++j) {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = A[k] + 2 * (A[k - 1] + A[k + 1]) + A[k - buf_stride] +
                        A[k - buf_stride - 1] + A[k - buf_stride + 1];
      const int32_t b = B[k] + 2 * (B[k - 1] + B[k + 1]) + B[k - buf_stride] +
                        B[k - buf_stride - 1] + B[k - buf_stride + 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
    j = width - 1;
    {
      const int k = i * buf_stride + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int nb = 3;
      const int32_t a = 3 * A[k] + 2 * A[k - 1] + 2 * A[k - buf_stride] +
                        A[k - buf_stride - 1];
      const int32_t b = 3 * B[k] + 2 * B[k - 1] + 2 * B[k - buf_stride] +
                        B[k - buf_stride - 1];
      const int32_t v = a * dgd[l] + b;
      dst[m] = ROUND_POWER_OF_TWO(v, SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    }
  }
}

void apply_selfguided_restoration_sse4_1(uint8_t *dat, int width, int height,
                                         int stride, int bit_depth, int eps,
                                         int *xqd, uint8_t *dst, int dst_stride,
                                         int32_t *tmpbuf) {
  int xq[2];
  int32_t *flt1 = tmpbuf;
  int32_t *flt2 = flt1 + RESTORATION_TILEPELS_MAX;
  int32_t *tmpbuf2 = flt2 + RESTORATION_TILEPELS_MAX;
  int i, j;
  assert(width * height <= RESTORATION_TILEPELS_MAX);
  // The SSE4.1 code currently only supports tiles which are a multiple of 4
  // pixels wide (but has no height restriction). If this is not the case,
  // we fall back to the C version.
  // Similarly, highbitdepth mode is not fully supported yet, so drop back
  // to the C code in that case.
  // TODO(david.barker): Allow non-multiple-of-4 widths and bit_depth > 8
  // in the SSE4.1 code.
  if ((width & 3) || bit_depth != 8) {
    apply_selfguided_restoration_c(dat, width, height, stride, bit_depth, eps,
                                   xqd, dst, dst_stride, tmpbuf);
    return;
  }
  av1_selfguided_restoration_sse4_1(dat, width, height, stride, flt1, width,
                                    bit_depth, sgr_params[eps].r1,
                                    sgr_params[eps].e1, tmpbuf2);
  av1_selfguided_restoration_sse4_1(dat, width, height, stride, flt2, width,
                                    bit_depth, sgr_params[eps].r2,
                                    sgr_params[eps].e2, tmpbuf2);
  decode_xq(xqd, xq);

  __m128i xq0 = _mm_set1_epi32(xq[0]);
  __m128i xq1 = _mm_set1_epi32(xq[1]);
  for (i = 0; i < height; ++i) {
    // Calculate output in batches of 8 pixels
    for (j = 0; j < width; j += 8) {
      const int k = i * width + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      __m128i src =
          _mm_slli_epi16(_mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)&dat[l])),
                         SGRPROJ_RST_BITS);

      const __m128i u_0 = _mm_cvtepu16_epi32(src);
      const __m128i u_1 = _mm_cvtepu16_epi32(_mm_srli_si128(src, 8));

      const __m128i f1_0 =
          _mm_sub_epi32(_mm_loadu_si128((__m128i *)&flt1[k]), u_0);
      const __m128i f2_0 =
          _mm_sub_epi32(_mm_loadu_si128((__m128i *)&flt2[k]), u_0);
      const __m128i f1_1 =
          _mm_sub_epi32(_mm_loadu_si128((__m128i *)&flt1[k + 4]), u_1);
      const __m128i f2_1 =
          _mm_sub_epi32(_mm_loadu_si128((__m128i *)&flt2[k + 4]), u_1);

      const __m128i v_0 = _mm_add_epi32(
          _mm_add_epi32(_mm_mullo_epi32(xq0, f1_0), _mm_mullo_epi32(xq1, f2_0)),
          _mm_slli_epi32(u_0, SGRPROJ_PRJ_BITS));
      const __m128i v_1 = _mm_add_epi32(
          _mm_add_epi32(_mm_mullo_epi32(xq0, f1_1), _mm_mullo_epi32(xq1, f2_1)),
          _mm_slli_epi32(u_1, SGRPROJ_PRJ_BITS));

      const __m128i rounding =
          _mm_set1_epi32((1 << (SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS)) >> 1);
      const __m128i w_0 = _mm_srai_epi32(_mm_add_epi32(v_0, rounding),
                                         SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
      const __m128i w_1 = _mm_srai_epi32(_mm_add_epi32(v_1, rounding),
                                         SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);

      const __m128i tmp = _mm_packs_epi32(w_0, w_1);
      const __m128i res = _mm_packus_epi16(tmp, tmp /* "don't care" value */);
      _mm_storel_epi64((__m128i *)&dst[m], res);
    }
    // Process leftover pixels
    for (; j < width; ++j) {
      const int k = i * width + j;
      const int l = i * stride + j;
      const int m = i * dst_stride + j;
      const int32_t u = ((int32_t)dat[l] << SGRPROJ_RST_BITS);
      const int32_t f1 = (int32_t)flt1[k] - u;
      const int32_t f2 = (int32_t)flt2[k] - u;
      const int32_t v = xq[0] * f1 + xq[1] * f2 + (u << SGRPROJ_PRJ_BITS);
      const int16_t w =
          (int16_t)ROUND_POWER_OF_TWO(v, SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
      dst[m] = (uint16_t)clip_pixel(w);
    }
  }
}
