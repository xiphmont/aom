#include "av1/common/daala_tx.h"
#include "av1/common/odintrin.h"

/* clang-format off */

# define OD_DCT_RSHIFT(_a, _b) OD_UNBIASED_RSHIFT32(_a, _b)

/* TODO: Daala DCT overflow checks need to be ported as a later test */
# if defined(OD_DCT_CHECK_OVERFLOW)
# else
#  define OD_DCT_OVERFLOW_CHECK(val, scale, offset, idx)
# endif

#define OD_FDCT_2_ASYM(p0, p1, p1h) \
  /* Embedded 2-point asymmetric Type-II fDCT. */ \
  do { \
    p0 += p1h; \
    p1 = p0 - p1; \
  } \
  while (0)

#define OD_IDCT_2_ASYM(p0, p1, p1h) \
  /* Embedded 2-point asymmetric Type-II iDCT. */ \
  do { \
    p1 = p0 - p1; \
    p1h = OD_DCT_RSHIFT(p1, 1); \
    p0 -= p1h; \
  } \
  while (0)

#define OD_FDST_2_ASYM(p0, p1) \
  /* Embedded 2-point asymmetric Type-IV fDST. */ \
  do { \
    /* 11507/16384 ~= 4*Sin[Pi/8] - 2*Tan[Pi/8] ~= 0.702306604714169 */ \
    OD_DCT_OVERFLOW_CHECK(p1, 11507, 8192, 187); \
    p0 -= (p1*11507 + 8192) >> 14; \
    /* 669/1024 ~= Cos[Pi/8]/Sqrt[2] ~= 0.653281482438188 */ \
    OD_DCT_OVERFLOW_CHECK(p0, 669, 512, 188); \
    p1 += (p0*669 + 512) >> 10; \
    /* 4573/4096 ~= 4*Sin[Pi/8] - Tan[Pi/8] ~= 1.11652016708726 */ \
    OD_DCT_OVERFLOW_CHECK(p1, 4573, 2048, 189); \
    p0 -= (p1*4573 + 2048) >> 12; \
  } \
  while (0)

#define OD_IDST_2_ASYM(p0, p1) \
  /* Embedded 2-point asymmetric Type-IV iDST. */ \
  do { \
    /* 4573/4096 ~= 4*Sin[Pi/8] - Tan[Pi/8] ~= 1.11652016708726 */ \
    p0 += (p1*4573 + 2048) >> 12; \
    /* 669/1024 ~= Cos[Pi/8]/Sqrt[2] ~= 0.653281482438188 */ \
    p1 -= (p0*669 + 512) >> 10; \
    /* 11507/16384 ~= 4*Sin[Pi/8] - 2*Tan[Pi/8] ~= 0.702306604714169 */ \
    p0 += (p1*11507 + 8192) >> 14; \
  } \
  while (0)

#define OD_FDCT_4(q0, q2, q1, q3) \
  /* Embedded 4-point orthonormal Type-II fDCT. */ \
  do { \
    int q2h; \
    int q3h; \
    q3 = q0 - q3; \
    q3h = OD_DCT_RSHIFT(q3, 1); \
    q0 -= q3h; \
    q2 += q1; \
    q2h = OD_DCT_RSHIFT(q2, 1); \
    q1 = q2h - q1; \
    OD_FDCT_2_ASYM(q0, q2, q2h); \
    OD_FDST_2_ASYM(q3, q1); \
  } \
  while (0)

#define OD_IDCT_4(q0, q2, q1, q3) \
  /* Embedded 4-point orthonormal Type-II iDCT. */ \
  do { \
    int q1h; \
    int q3h; \
    OD_IDST_2_ASYM(q3, q2); \
    OD_IDCT_2_ASYM(q0, q1, q1h); \
    q3h = OD_DCT_RSHIFT(q3, 1); \
    q0 += q3h; \
    q3 = q0 - q3; \
    q2 = q1h - q2; \
    q1 -= q2; \
  } \
  while (0)

#define OD_FDST_4(q0, q2, q1, q3) \
  /* Embedded 4-point orthonormal Type-IV fDST. */ \
  do { \
    int q0h; \
    int q1h; \
    /* 13573/32768 ~= Tan[Pi/8] ~= 0.414213562373095 */ \
    OD_DCT_OVERFLOW_CHECK(q1, 13573, 16384, 190); \
    q2 += (q1*13573 + 16384) >> 15; \
    /* 5793/8192 ~= Sin[Pi/4] ~= 0.707106781186547 */ \
    OD_DCT_OVERFLOW_CHECK(q2, 5793, 4096, 191); \
    q1 -= (q2*5793 + 4096) >> 13; \
    /* 3393/8192 ~= Tan[Pi/8] ~= 0.414213562373095 */ \
    OD_DCT_OVERFLOW_CHECK(q1, 3393, 4096, 192); \
    q2 += (q1*3393 + 4096) >> 13; \
    q0 += q2; \
    q0h = OD_DCT_RSHIFT(q0, 1); \
    q2 = q0h - q2; \
    q1 += q3; \
    q1h = OD_DCT_RSHIFT(q1, 1); \
    q3 -= q1h; \
    /* 537/1024 ~= (1/Sqrt[2] - Cos[3*Pi/16]/2)/Sin[3*Pi/16] ~=
        0.524455699240090 */ \
    OD_DCT_OVERFLOW_CHECK(q1, 537, 512, 193); \
    q2 -= (q1*537 + 512) >> 10; \
    /* 1609/2048 ~= Sqrt[2]*Sin[3*Pi/16] ~= 0.785694958387102 */ \
    OD_DCT_OVERFLOW_CHECK(q2, 1609, 1024, 194); \
    q1 += (q2*1609 + 1024) >> 11; \
    /* 7335/32768 ~= (1/Sqrt[2] - Cos[3*Pi/16])/Sin[3*Pi/16] ~=
        0.223847182092655 */ \
    OD_DCT_OVERFLOW_CHECK(q1, 7335, 16384, 195); \
    q2 += (q1*7335 + 16384) >> 15; \
    /* 5091/8192 ~= (1/Sqrt[2] - Cos[7*Pi/16]/2)/Sin[7*Pi/16] ~=
        0.6215036383171189 */ \
    OD_DCT_OVERFLOW_CHECK(q0, 5091, 4096, 196); \
    q3 += (q0*5091 + 4096) >> 13; \
    /* 5681/4096 ~= Sqrt[2]*Sin[7*Pi/16] ~= 1.38703984532215 */ \
    OD_DCT_OVERFLOW_CHECK(q3, 5681, 2048, 197); \
    q0 -= (q3*5681 + 2048) >> 12; \
    /* 4277/8192 ~= (1/Sqrt[2] - Cos[7*Pi/16])/Sin[7*Pi/16] ~=
        0.52204745462729 */ \
    OD_DCT_OVERFLOW_CHECK(q0, 4277, 4096, 198); \
    q3 += (q0*4277 + 4096) >> 13; \
  } \
  while (0)

#define OD_IDST_4(q0, q2, q1, q3) \
  /* Embedded 4-point orthonormal Type-IV iDST. */ \
  do { \
    int q0h; \
    int q2h; \
    /* 4277/8192 ~= (1/Sqrt[2] - Cos[7*Pi/16])/Sin[7*Pi/16] ~=
        0.52204745462729 */ \
    q3 -= (q0*4277 + 4096) >> 13; \
    /* 5681/4096 ~= Sqrt[2]*Sin[7*Pi/16] ~= 1.38703984532215 */ \
    q0 += (q3*5681 + 2048) >> 12; \
    /* 5091/8192 ~= (1/Sqrt[2] - Cos[7*Pi/16]/2)/Sin[7*Pi/16] ~=
        0.6215036383171189 */ \
    q3 -= (q0*5091 + 4096) >> 13; \
    /* 7335/32768 ~= (1/Sqrt[2] - Cos[3*Pi/16])/Sin[3*Pi/16] ~=
        0.223847182092655 */ \
    q1 -= (q2*7335 + 16384) >> 15; \
    /* 1609/2048 ~= Sqrt[2]*Sin[3*Pi/16] ~= 0.785694958387102 */ \
    q2 -= (q1*1609 + 1024) >> 11; \
    /* 537/1024 ~= (1/Sqrt[2] - Cos[3*Pi/16]/2)/Sin[3*Pi/16] ~=
        0.524455699240090 */ \
    q1 += (q2*537 + 512) >> 10; \
    q2h = OD_DCT_RSHIFT(q2, 1); \
    q3 += q2h; \
    q2 -= q3; \
    q0h = OD_DCT_RSHIFT(q0, 1); \
    q1 = q0h - q1; \
    q0 -= q1; \
    /* 3393/8192 ~= Tan[Pi/8] ~= 0.414213562373095 */ \
    q1 -= (q2*3393 + 4096) >> 13; \
    /* 5793/8192 ~= Sin[Pi/4] ~= 0.707106781186547 */ \
    q2 += (q1*5793 + 4096) >> 13; \
    /* 13573/32768 ~= Tan[Pi/8] ~= 0.414213562373095 */ \
    q1 -= (q2*13573 + 16384) >> 15; \
  } \
  while (0)

void od_bin_fdct4(od_coeff y[4], const od_coeff *x, int xstride) {
  int q0;
  int q1;
  int q2;
  int q3;
  q0 = x[0*xstride];
  q2 = x[1*xstride];
  q1 = x[2*xstride];
  q3 = x[3*xstride];
  OD_FDCT_4(q0, q2, q1, q3);
  y[0] = (od_coeff)q0;
  y[1] = (od_coeff)q1;
  y[2] = (od_coeff)q2;
  y[3] = (od_coeff)q3;
}

void od_bin_idct4(od_coeff *x, int xstride, const od_coeff y[4]) {
  int q0;
  int q1;
  int q2;
  int q3;
  q0 = y[0];
  q2 = y[1];
  q1 = y[2];
  q3 = y[3];
  OD_IDCT_4(q0, q2, q1, q3);
  x[0*xstride] = q0;
  x[1*xstride] = q1;
  x[2*xstride] = q2;
  x[3*xstride] = q3;
}


void od_bin_fdst4(od_coeff y[4], const od_coeff *x, int xstride) {
  int q0;
  int q1;
  int q2;
  int q3;
  q0 = x[3*xstride];
  q2 = x[2*xstride];
  q1 = x[1*xstride];
  q3 = x[0*xstride];
  OD_FDST_4(q0, q2, q1, q3);
  y[0] = (od_coeff)q3;
  y[1] = (od_coeff)q2;
  y[2] = (od_coeff)q1;
  y[3] = (od_coeff)q0;

  // If we want to make it more like the stock AV1 ADST, use the following...
  //y[0] = (od_coeff)q3;
  //y[1] = (od_coeff)-q2;
  //y[2] = (od_coeff)q1;
  //y[3] = (od_coeff)-q0;
}

void od_bin_idst4(od_coeff *x, int xstride, const od_coeff y[4]) {
  int q0;
  int q1;
  int q2;
  int q3;
  q0 = y[3];
  q2 = y[2];
  q1 = y[1];
  q3 = y[0];
  // If we want to make it more like the stock AV1 ADST, use the following...
  //q0 = -y[3];
  //q2 = y[2];
  //q1 = -y[1];
  //q3 = y[0];

  OD_IDST_4(q0, q2, q1, q3);
  x[0*xstride] = q3;
  x[1*xstride] = q2;
  x[2*xstride] = q1;
  x[3*xstride] = q0;
}
