/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AV1_COMMON_RESTORATION_H_
#define AV1_COMMON_RESTORATION_H_

#include "aom_ports/mem.h"
#include "./aom_config.h"

#include "av1/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CLIP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))
#define RINT(x) ((x) < 0 ? (int)((x)-0.5) : (int)((x) + 0.5))

#define RESTORATION_TILESIZE_SML 128
#define RESTORATION_TILESIZE_BIG 256
#define RESTORATION_TILEPELS_MAX \
  (RESTORATION_TILESIZE_BIG * RESTORATION_TILESIZE_BIG * 9 / 4)

#if USE_DOMAINTXFMRF
#define DOMAINTXFMRF_PARAMS_BITS 6
#define DOMAINTXFMRF_PARAMS (1 << DOMAINTXFMRF_PARAMS_BITS)
#define DOMAINTXFMRF_SIGMA_SCALEBITS 4
#define DOMAINTXFMRF_SIGMA_SCALE (1 << DOMAINTXFMRF_SIGMA_SCALEBITS)
#define DOMAINTXFMRF_ITERS 3
#define DOMAINTXFMRF_VTABLE_PRECBITS 8
#define DOMAINTXFMRF_VTABLE_PREC (1 << DOMAINTXFMRF_VTABLE_PRECBITS)
#define DOMAINTXFMRF_MULT \
  sqrt(((1 << (DOMAINTXFMRF_ITERS * 2)) - 1) * 2.0 / 3.0)
// 1 32-bit and 2 8-bit buffers needed for the filter
#define DOMAINTXFMRF_TMPBUF_SIZE \
  (RESTORATION_TILEPELS_MAX * (sizeof(int32_t) + 2 * sizeof(uint8_t)))
// One extra buffer needed in encoder, which is either 8-bit or 16-bit
// depending on the video bit depth.
#if CONFIG_AOM_HIGHBITDEPTH
#define DOMAINTXFMRF_EXTBUF_SIZE (RESTORATION_TILEPELS_MAX * sizeof(uint16_t))
#else
#define DOMAINTXFMRF_EXTBUF_SIZE (RESTORATION_TILEPELS_MAX * sizeof(uint8_t))
#endif
#define DOMAINTXFMRF_BITS (DOMAINTXFMRF_PARAMS_BITS)
#endif  // USE_DOMAINTXFMRF

// 4 32-bit buffers needed for the filter:
// 2 for the restored versions of the frame and
// 2 for each restoration operation
#define SGRPROJ_OUTBUF_SIZE \
  ((RESTORATION_TILESIZE_BIG * 3 / 2) * (RESTORATION_TILESIZE_BIG * 3 / 2 + 16))
#define SGRPROJ_TMPBUF_SIZE                         \
  (RESTORATION_TILEPELS_MAX * 2 * sizeof(int32_t) + \
   SGRPROJ_OUTBUF_SIZE * 2 * sizeof(int32_t))
#define SGRPROJ_EXTBUF_SIZE (0)
#define SGRPROJ_PARAMS_BITS 4
#define SGRPROJ_PARAMS (1 << SGRPROJ_PARAMS_BITS)

// Precision bits for projection
#define SGRPROJ_PRJ_BITS 7
// Restoration precision bits generated higher than source before projection
#define SGRPROJ_RST_BITS 4
// Internal precision bits for core selfguided_restoration
#define SGRPROJ_SGR_BITS 8
#define SGRPROJ_SGR (1 << SGRPROJ_SGR_BITS)

#define SGRPROJ_PRJ_MIN0 (-(1 << SGRPROJ_PRJ_BITS) / 4)
#define SGRPROJ_PRJ_MAX0 (SGRPROJ_PRJ_MIN0 + (1 << SGRPROJ_PRJ_BITS) - 1)
#define SGRPROJ_PRJ_MIN1 (-(1 << SGRPROJ_PRJ_BITS) / 4)
#define SGRPROJ_PRJ_MAX1 (SGRPROJ_PRJ_MIN1 + (1 << SGRPROJ_PRJ_BITS) - 1)

#define SGRPROJ_BITS (SGRPROJ_PRJ_BITS * 2 + SGRPROJ_PARAMS_BITS)

#define MAX_RADIUS 3  // Only 1, 2, 3 allowed
#define MAX_EPS 80    // Max value of eps
#define MAX_NELEM ((2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1))
#define SGRPROJ_MTABLE_BITS 20
#define SGRPROJ_RECIP_BITS 12

#define WIENER_HALFWIN 3
#define WIENER_HALFWIN1 (WIENER_HALFWIN + 1)
#define WIENER_WIN (2 * WIENER_HALFWIN + 1)
#define WIENER_WIN2 ((WIENER_WIN) * (WIENER_WIN))
#define WIENER_TMPBUF_SIZE (0)
#define WIENER_EXTBUF_SIZE (0)

#define WIENER_FILT_PREC_BITS 7
#define WIENER_FILT_STEP (1 << WIENER_FILT_PREC_BITS)

// Central values for the taps
#define WIENER_FILT_TAP0_MIDV (3)
#define WIENER_FILT_TAP1_MIDV (-7)
#define WIENER_FILT_TAP2_MIDV (15)

#define WIENER_FILT_TAP0_BITS 4
#define WIENER_FILT_TAP1_BITS 5
#define WIENER_FILT_TAP2_BITS 6

#define WIENER_FILT_BITS \
  ((WIENER_FILT_TAP0_BITS + WIENER_FILT_TAP1_BITS + WIENER_FILT_TAP2_BITS) * 2)

#define WIENER_FILT_TAP0_MINV \
  (WIENER_FILT_TAP0_MIDV - (1 << WIENER_FILT_TAP0_BITS) / 2)
#define WIENER_FILT_TAP1_MINV \
  (WIENER_FILT_TAP1_MIDV - (1 << WIENER_FILT_TAP1_BITS) / 2)
#define WIENER_FILT_TAP2_MINV \
  (WIENER_FILT_TAP2_MIDV - (1 << WIENER_FILT_TAP2_BITS) / 2)

#define WIENER_FILT_TAP0_MAXV \
  (WIENER_FILT_TAP0_MIDV - 1 + (1 << WIENER_FILT_TAP0_BITS) / 2)
#define WIENER_FILT_TAP1_MAXV \
  (WIENER_FILT_TAP1_MIDV - 1 + (1 << WIENER_FILT_TAP1_BITS) / 2)
#define WIENER_FILT_TAP2_MAXV \
  (WIENER_FILT_TAP2_MIDV - 1 + (1 << WIENER_FILT_TAP2_BITS) / 2)

// Max of SGRPROJ_TMPBUF_SIZE, DOMAINTXFMRF_TMPBUF_SIZE, WIENER_TMPBUF_SIZE
#define RESTORATION_TMPBUF_SIZE (SGRPROJ_TMPBUF_SIZE)

#if USE_DOMAINTXFMRF
// Max of SGRPROJ_EXTBUF_SIZE, DOMAINTXFMRF_EXTBUF_SIZE, WIENER_EXTBUF_SIZE
#define RESTORATION_EXTBUF_SIZE (DOMAINTXFMRF_EXTBUF_SIZE)
#else
#define RESTORATION_EXTBUF_SIZE (WIENER_EXTBUF_SIZE)
#endif  // USE_DOMAINTXFMRF

// Check the assumptions of the existing code
#if SUBPEL_TAPS != WIENER_WIN + 1
#error "Wiener filter currently only works if SUBPEL_TAPS == WIENER_WIN + 1"
#endif
#if WIENER_FILT_PREC_BITS != 7
#error "Wiener filter currently only works if WIENER_FILT_PREC_BITS == 7"
#endif
typedef struct {
  DECLARE_ALIGNED(16, InterpKernel, vfilter);
  DECLARE_ALIGNED(16, InterpKernel, hfilter);
} WienerInfo;

typedef struct {
  int r1;
  int e1;
  int r2;
  int e2;
} sgr_params_type;

typedef struct {
  int ep;
  int xqd[2];
} SgrprojInfo;

#if USE_DOMAINTXFMRF
typedef struct { int sigma_r; } DomaintxfmrfInfo;
#endif  // USE_DOMAINTXFMRF

typedef struct {
  RestorationType frame_restoration_type;
  RestorationType *restoration_type;
  // Wiener filter
  WienerInfo *wiener_info;
  // Selfguided proj filter
  SgrprojInfo *sgrproj_info;
#if USE_DOMAINTXFMRF
  // Domain transform filter
  DomaintxfmrfInfo *domaintxfmrf_info;
#endif  // USE_DOMAINTXFMRF
} RestorationInfo;

typedef struct {
  RestorationInfo *rsi;
  int keyframe;
  int ntiles;
  int tile_width, tile_height;
  int nhtiles, nvtiles;
  int32_t *tmpbuf;
} RestorationInternal;

static INLINE int get_rest_tilesize(int width, int height) {
  if (width * height <= 352 * 288)
    return RESTORATION_TILESIZE_SML;
  else
    return RESTORATION_TILESIZE_BIG;
}

static INLINE int av1_get_rest_ntiles(int width, int height, int *tile_width,
                                      int *tile_height, int *nhtiles,
                                      int *nvtiles) {
  int nhtiles_, nvtiles_;
  int tile_width_, tile_height_;
  int tilesize = get_rest_tilesize(width, height);
  tile_width_ = (tilesize < 0) ? width : AOMMIN(tilesize, width);
  tile_height_ = (tilesize < 0) ? height : AOMMIN(tilesize, height);
  nhtiles_ = (width + (tile_width_ >> 1)) / tile_width_;
  nvtiles_ = (height + (tile_height_ >> 1)) / tile_height_;
  if (tile_width) *tile_width = tile_width_;
  if (tile_height) *tile_height = tile_height_;
  if (nhtiles) *nhtiles = nhtiles_;
  if (nvtiles) *nvtiles = nvtiles_;
  return (nhtiles_ * nvtiles_);
}

static INLINE void av1_get_rest_tile_limits(
    int tile_idx, int subtile_idx, int subtile_bits, int nhtiles, int nvtiles,
    int tile_width, int tile_height, int im_width, int im_height, int clamp_h,
    int clamp_v, int *h_start, int *h_end, int *v_start, int *v_end) {
  const int htile_idx = tile_idx % nhtiles;
  const int vtile_idx = tile_idx / nhtiles;
  *h_start = htile_idx * tile_width;
  *v_start = vtile_idx * tile_height;
  *h_end = (htile_idx < nhtiles - 1) ? *h_start + tile_width : im_width;
  *v_end = (vtile_idx < nvtiles - 1) ? *v_start + tile_height : im_height;
  if (subtile_bits) {
    const int num_subtiles_1d = (1 << subtile_bits);
    const int subtile_width = (*h_end - *h_start) >> subtile_bits;
    const int subtile_height = (*v_end - *v_start) >> subtile_bits;
    const int subtile_idx_h = subtile_idx & (num_subtiles_1d - 1);
    const int subtile_idx_v = subtile_idx >> subtile_bits;
    *h_start += subtile_idx_h * subtile_width;
    *v_start += subtile_idx_v * subtile_height;
    *h_end = subtile_idx_h == num_subtiles_1d - 1 ? *h_end
                                                  : *h_start + subtile_width;
    *v_end = subtile_idx_v == num_subtiles_1d - 1 ? *v_end
                                                  : *v_start + subtile_height;
  }
  if (clamp_h) {
    *h_start = AOMMAX(*h_start, clamp_h);
    *h_end = AOMMIN(*h_end, im_width - clamp_h);
  }
  if (clamp_v) {
    *v_start = AOMMAX(*v_start, clamp_v);
    *v_end = AOMMIN(*v_end, im_height - clamp_v);
  }
}

extern const sgr_params_type sgr_params[SGRPROJ_PARAMS];
extern int sgrproj_mtable[MAX_EPS][MAX_NELEM];
extern const int32_t x_by_xplus1[256];
extern const int32_t one_by_x[MAX_NELEM];

int av1_alloc_restoration_struct(struct AV1Common *cm,
                                 RestorationInfo *rst_info, int width,
                                 int height);
void av1_free_restoration_struct(RestorationInfo *rst_info);

void extend_frame(uint8_t *data, int width, int height, int stride);
void av1_selfguided_restoration(int32_t *dgd, int width, int height, int stride,
                                int bit_depth, int r, int eps, int32_t *tmpbuf);
#if USE_DOMAINTXFMRF
void av1_domaintxfmrf_restoration(uint8_t *dgd, int width, int height,
                                  int stride, int param, uint8_t *dst,
                                  int dst_stride, int32_t *tmpbuf);
#endif  // USE_DOMAINTXFMRF
#if CONFIG_AOM_HIGHBITDEPTH
void extend_frame_highbd(uint16_t *data, int width, int height, int stride);
#if USE_DOMAINTXFMRF
void av1_domaintxfmrf_restoration_highbd(uint16_t *dgd, int width, int height,
                                         int stride, int param, int bit_depth,
                                         uint16_t *dst, int dst_stride,
                                         int32_t *tmpbuf);
#endif  // USE_DOMAINTXFMRF
#endif  // CONFIG_AOM_HIGHBITDEPTH
void decode_xq(int *xqd, int *xq);
void av1_loop_restoration_frame(YV12_BUFFER_CONFIG *frame, struct AV1Common *cm,
                                RestorationInfo *rsi, int components_pattern,
                                int partial_frame, YV12_BUFFER_CONFIG *dst);
void av1_loop_restoration_precal();
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AV1_COMMON_RESTORATION_H_
