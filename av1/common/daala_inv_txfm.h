/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AV1_ENCODER_DAALA_INV_TXFM_H_
#define AV1_ENCODER_DAALA_INV_TXFM_H_

#include "./aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif

void daala_inv_txfm_add(const tran_low_t *input_coeffs, void *output_pixels,
                        int output_stride, TxfmParam *txfm_param, int hbd);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AV1_ENCODER_DAALA_INV_TXFM_H_
