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

#ifndef AV1_ENCODER_HYBRID_FWD_TXFM_H_
#define AV1_ENCODER_HYBRID_FWD_TXFM_H_

#include "./aom_config.h"
#include "./aom_dsp_rtcd.h"

typedef enum FWD_TXFM_OPT { FWD_TXFM_OPT_NORMAL } FWD_TXFM_OPT;

typedef struct FWD_TXFM_PARAM {
  TX_TYPE tx_type;
  TX_SIZE tx_size;
  int lossless;
#if CONFIG_AOM_HIGHBITDEPTH
  int bd;
#endif  // CONFIG_AOM_HIGHBITDEPTH
} FWD_TXFM_PARAM;

#ifdef __cplusplus
extern "C" {
#endif

void fwd_txfm(const int16_t *src_diff, tran_low_t *coeff, int diff_stride,
              FWD_TXFM_PARAM *fwd_txfm_param);

#if CONFIG_AOM_HIGHBITDEPTH
void highbd_fwd_txfm(const int16_t *src_diff, tran_low_t *coeff,
                     int diff_stride, FWD_TXFM_PARAM *fwd_txfm_param);
#endif  // CONFIG_AOM_HIGHBITDEPTH

static INLINE int get_tx1d_size(TX_SIZE tx_size) {
  switch (tx_size) {
    case TX_32X32: return 32;
    case TX_16X16: return 16;
    case TX_8X8: return 8;
    case TX_4X4: return 4;
    default: assert(0); return -1;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AV1_ENCODER_HYBRID_FWD_TXFM_H_
