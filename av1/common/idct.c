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

#include <math.h>

#include "./aom_dsp_rtcd.h"
#include "./av1_rtcd.h"
#include "aom_dsp/inv_txfm.h"
#include "aom_ports/mem.h"
#include "av1/common/av1_inv_txfm1d_cfg.h"
#include "av1/common/blockd.h"
#include "av1/common/enums.h"
#include "av1/common/idct.h"
#if CONFIG_DAALA_DCT4 || CONFIG_DAALA_DCT8 || CONFIG_DAALA_DCT16 || \
    CONFIG_DAALA_DCT32 || CONFIG_DAALA_DCT64
#include "av1/common/daala_tx.h"
#endif

int av1_get_tx_scale(const TX_SIZE tx_size) {
  //const int pels = tx_size_2d[tx_size];
  (void)tx_size;
  return 0;//(pels > 256) + (pels > 1024) + (pels > 4096);
}

// NOTE: The implementation of all inverses need to be aware of the fact
// that input and output could be the same buffer.

#if CONFIG_EXT_TX
#if !CONFIG_DAALA_DCT4 || !CONFIG_DAALA_DCT8 || !CONFIG_DAALA_DCT16
static void iidtx4_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  for (i = 0; i < 4; ++i) {
    output[i] = (tran_low_t)dct_const_round_shift(input[i] * Sqrt2);
  }
}
#endif

#if !CONFIG_DAALA_DCT4 || !CONFIG_DAALA_DCT8 || !CONFIG_DAALA_DCT16 || \
    !CONFIG_DAALA_DCT32
static void iidtx8_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  for (i = 0; i < 8; ++i) {
    output[i] = input[i] * 2;
  }
}

static void iidtx16_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  for (i = 0; i < 16; ++i) {
    output[i] = (tran_low_t)dct_const_round_shift(input[i] * 2 * Sqrt2);
  }
}
#endif

#if !CONFIG_DAALA_DCT8 || !CONFIG_DAALA_DCT16 || !CONFIG_DAALA_DCT32
static void iidtx32_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  for (i = 0; i < 32; ++i) {
    output[i] = input[i] * 4;
  }
}
#endif

#if CONFIG_TX64X64 && !CONFIG_DAALA_DCT64
static void iidtx64_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  for (i = 0; i < 64; ++i) {
    output[i] = (tran_low_t)dct_const_round_shift(input[i] * 4 * Sqrt2);
  }
}
#endif  // CONFIG_TX64X64
#endif  // CONFIG_EXT_TX

#if !CONFIG_DAALA_DCT8 || !CONFIG_DAALA_DCT16 || !CONFIG_DAALA_DCT32
// For use in lieu of ADST
static void ihalfright32_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  tran_low_t inputhalf[16];
  // Multiply input by sqrt(2)
  for (i = 0; i < 16; ++i) {
    inputhalf[i] = (tran_low_t)dct_const_round_shift(input[i] * Sqrt2);
  }
  for (i = 0; i < 16; ++i) {
    output[i] = input[16 + i] * 4;
  }
  aom_idct16_c(inputhalf, output + 16);
  // Note overall scaling factor is 4 times orthogonal
}
#endif

#if CONFIG_TX64X64 && !CONFIG_DAALA_DCT64
static void idct64_col_c(const tran_low_t *input, tran_low_t *output) {
  int32_t in[64], out[64];
  int i;
  for (i = 0; i < 64; ++i) in[i] = (int32_t)input[i];
  av1_idct64_new(in, out, inv_cos_bit_col_dct_64, inv_stage_range_col_dct_64);
  for (i = 0; i < 64; ++i) output[i] = (tran_low_t)out[i];
}

static void idct64_row_c(const tran_low_t *input, tran_low_t *output) {
  int32_t in[64], out[64];
  int i;
  for (i = 0; i < 64; ++i) in[i] = (int32_t)input[i];
  av1_idct64_new(in, out, inv_cos_bit_row_dct_64, inv_stage_range_row_dct_64);
  for (i = 0; i < 64; ++i) output[i] = (tran_low_t)out[i];
}

// For use in lieu of ADST
static void ihalfright64_c(const tran_low_t *input, tran_low_t *output) {
  int i;
  tran_low_t inputhalf[32];
  // Multiply input by sqrt(2)
  for (i = 0; i < 32; ++i) {
    inputhalf[i] = (tran_low_t)dct_const_round_shift(input[i] * Sqrt2);
  }
  for (i = 0; i < 32; ++i) {
    output[i] = (tran_low_t)dct_const_round_shift(input[32 + i] * 4 * Sqrt2);
  }
  aom_idct32_c(inputhalf, output + 32);
  // Note overall scaling factor is 4 * sqrt(2)  times orthogonal
}
#endif  // CONFIG_TX64X64

#define FLIPUD_PTR(dest, stride, size)       \
  do {                                       \
    (dest) = (dest) + ((size)-1) * (stride); \
    (stride) = -(stride);                    \
  } while (0)

#if CONFIG_EXT_TX
static void maybe_flip_strides(uint8_t **dst, int *dstride, tran_low_t **src,
                               int *sstride, int tx_type, int sizey,
                               int sizex) {
  // Note that the transpose of src will be added to dst. In order to LR
  // flip the addends (in dst coordinates), we UD flip the src. To UD flip
  // the addends, we UD flip the dst.
  switch (tx_type) {
    case DCT_DCT:
    case ADST_DCT:
    case DCT_ADST:
    case ADST_ADST:
    case IDTX:
    case V_DCT:
    case H_DCT:
    case V_ADST:
    case H_ADST: break;
    case FLIPADST_DCT:
    case FLIPADST_ADST:
    case V_FLIPADST:
      // flip UD
      FLIPUD_PTR(*dst, *dstride, sizey);
      break;
    case DCT_FLIPADST:
    case ADST_FLIPADST:
    case H_FLIPADST:
      // flip LR
      FLIPUD_PTR(*src, *sstride, sizex);
      break;
    case FLIPADST_FLIPADST:
      // flip UD
      FLIPUD_PTR(*dst, *dstride, sizey);
      // flip LR
      FLIPUD_PTR(*src, *sstride, sizex);
      break;
    default: assert(0); break;
  }
}
#endif  // CONFIG_EXT_TX

#if CONFIG_HIGHBITDEPTH
#if CONFIG_EXT_TX && CONFIG_TX64X64
static void highbd_inv_idtx_add_c(const tran_low_t *input, uint8_t *dest8,
                                  int stride, int bs, int tx_type, int bd) {
  int r, c;
  const int shift = bs < 32 ? 3 : 2;
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);

  if (tx_type == IDTX) {
    for (r = 0; r < bs; ++r) {
      for (c = 0; c < bs; ++c)
        dest[c] = highbd_clip_pixel_add(dest[c], input[c] >> shift, bd);
      dest += stride;
      input += bs;
    }
  }
}
#endif  // CONFIG_EXT_TX && CONFIG_TX64X64
#endif  // CONFIG_HIGHBITDEPTH

#if CONFIG_LGT
void ilgt4(const tran_low_t *input, tran_low_t *output,
           const tran_high_t *lgtmtx) {
  if (!lgtmtx) assert(0);
  // evaluate s[j] = sum of all lgtmtx[j]*input[i] over i=1,...,4
  tran_high_t s[4] = { 0 };
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) s[j] += lgtmtx[i * 4 + j] * input[i];

  for (int i = 0; i < 4; ++i) output[i] = WRAPLOW(dct_const_round_shift(s[i]));
}

void ilgt8(const tran_low_t *input, tran_low_t *output,
           const tran_high_t *lgtmtx) {
  if (!lgtmtx) assert(0);
  // evaluate s[j] = sum of all lgtmtx[j]*input[i] over i=1,...,8
  tran_high_t s[8] = { 0 };
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) s[j] += lgtmtx[i * 8 + j] * input[i];

  for (int i = 0; i < 8; ++i) output[i] = WRAPLOW(dct_const_round_shift(s[i]));
}

// get_lgt4 and get_lgt8 return 1 and pick a lgt matrix if LGT is chosen to
// apply. Otherwise they return 0
int get_lgt4(const TxfmParam *txfm_param, int is_col,
             const tran_high_t **lgtmtx) {
  if (is_col && (vtx_tab[txfm_param->tx_type] == ADST_1D ||
                 vtx_tab[txfm_param->tx_type] == FLIPADST_1D)) {
    lgtmtx[0] = txfm_param->is_inter ? &lgt4_170[0][0] : &lgt4_140[0][0];
    return 1;
  } else if (!is_col && (htx_tab[txfm_param->tx_type] == ADST_1D ||
                         htx_tab[txfm_param->tx_type] == FLIPADST_1D)) {
    lgtmtx[0] = txfm_param->is_inter ? &lgt4_170[0][0] : &lgt4_140[0][0];
    return 1;
  }
  lgtmtx[0] = NULL;
  return 0;
}

int get_lgt8(const TxfmParam *txfm_param, int is_col,
             const tran_high_t **lgtmtx) {
  if (is_col && (vtx_tab[txfm_param->tx_type] == ADST_1D ||
                 vtx_tab[txfm_param->tx_type] == FLIPADST_1D)) {
    lgtmtx[0] = txfm_param->is_inter ? &lgt8_170[0][0] : &lgt8_150[0][0];
    return 1;
  } else if (!is_col && (htx_tab[txfm_param->tx_type] == ADST_1D ||
                         htx_tab[txfm_param->tx_type] == FLIPADST_1D)) {
    lgtmtx[0] = txfm_param->is_inter ? &lgt8_170[0][0] : &lgt8_150[0][0];
    return 1;
  }
  lgtmtx[0] = NULL;
  return 0;
}
#endif  // CONFIG_LGT

void av1_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                         const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if !CONFIG_DAALA_DCT4
  if (tx_type == DCT_DCT) {
    aom_idct4x4_16_add(input, dest, stride);
    return;
  }
#endif
  static const transform_2d IHT_4[] = {
#if CONFIG_DAALA_DCT4
    { daala_idct4, daala_idct4 },  // DCT_DCT  = 0
    { daala_idst4, daala_idct4 },  // ADST_DCT = 1
    { daala_idct4, daala_idst4 },  // DCT_ADST = 2
    { daala_idst4, daala_idst4 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst4, daala_idct4 },  // FLIPADST_DCT
    { daala_idct4, daala_idst4 },  // DCT_FLIPADST
    { daala_idst4, daala_idst4 },  // FLIPADST_FLIPADST
    { daala_idst4, daala_idst4 },  // ADST_FLIPADST
    { daala_idst4, daala_idst4 },  // FLIPADST_ADST
    { daala_idtx4, daala_idtx4 },  // IDTX
    { daala_idct4, daala_idtx4 },  // V_DCT
    { daala_idtx4, daala_idct4 },  // H_DCT
    { daala_idst4, daala_idtx4 },  // V_ADST
    { daala_idtx4, daala_idst4 },  // H_ADST
    { daala_idst4, daala_idtx4 },  // V_FLIPADST
    { daala_idtx4, daala_idst4 },  // H_FLIPADST
#endif
#else
    { aom_idct4_c, aom_idct4_c },    // DCT_DCT  = 0
    { aom_iadst4_c, aom_idct4_c },   // ADST_DCT = 1
    { aom_idct4_c, aom_iadst4_c },   // DCT_ADST = 2
    { aom_iadst4_c, aom_iadst4_c },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { aom_iadst4_c, aom_idct4_c },   // FLIPADST_DCT
    { aom_idct4_c, aom_iadst4_c },   // DCT_FLIPADST
    { aom_iadst4_c, aom_iadst4_c },  // FLIPADST_FLIPADST
    { aom_iadst4_c, aom_iadst4_c },  // ADST_FLIPADST
    { aom_iadst4_c, aom_iadst4_c },  // FLIPADST_ADST
    { iidtx4_c, iidtx4_c },          // IDTX
    { aom_idct4_c, iidtx4_c },       // V_DCT
    { iidtx4_c, aom_idct4_c },       // H_DCT
    { aom_iadst4_c, iidtx4_c },      // V_ADST
    { iidtx4_c, aom_iadst4_c },      // H_ADST
    { aom_iadst4_c, iidtx4_c },      // V_FLIPADST
    { iidtx4_c, aom_iadst4_c },      // H_FLIPADST
#endif
#endif
  };

  int i, j;
  tran_low_t tmp[4][4];
  tran_low_t out[4][4];
  tran_low_t *outp = &out[0][0];
  int outstride = 4;

#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_col = get_lgt4(txfm_param, 1, lgtmtx_col);
  int use_lgt_row = get_lgt4(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors
  for (i = 0; i < 4; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt4(input, out[i], lgtmtx_row[0]);
    else
#endif
      IHT_4[tx_type].rows(input, out[i]);
    input += 4;
  }

  // transpose
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 4; j++) {
      tmp[j][i] = out[i][j];
    }
  }

  // inverse transform column vectors
  for (i = 0; i < 4; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt4(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_4[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, 4, 4);
#endif

  // Sum with the destination
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT4
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 4));
#endif
    }
  }
}

void av1_iht4x8_32_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                         const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_4x8[] = {
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT8
    { daala_idct8, daala_idct4 },  // DCT_DCT  = 0
    { daala_idst8, daala_idct4 },  // ADST_DCT = 1
    { daala_idct8, daala_idst4 },  // DCT_ADST = 2
    { daala_idst8, daala_idst4 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst8, daala_idct4 },  // FLIPADST_DCT
    { daala_idct8, daala_idst4 },  // DCT_FLIPADST
    { daala_idst8, daala_idst4 },  // FLIPADST_FLIPADST
    { daala_idst8, daala_idst4 },  // ADST_FLIPADST
    { daala_idst8, daala_idst4 },  // FLIPADST_ADST
    { daala_idtx8, daala_idtx4 },  // IDTX
    { daala_idct8, daala_idtx4 },  // V_DCT
    { daala_idtx8, daala_idct4 },  // H_DCT
    { daala_idst8, daala_idtx4 },  // V_ADST
    { daala_idtx8, daala_idst4 },  // H_ADST
    { daala_idst8, daala_idtx4 },  // V_FLIPADST
    { daala_idtx8, daala_idst4 },  // H_FLIPADST
#endif
#else
    { aom_idct8_c, aom_idct4_c },    // DCT_DCT
    { aom_iadst8_c, aom_idct4_c },   // ADST_DCT
    { aom_idct8_c, aom_iadst4_c },   // DCT_ADST
    { aom_iadst8_c, aom_iadst4_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst8_c, aom_idct4_c },   // FLIPADST_DCT
    { aom_idct8_c, aom_iadst4_c },   // DCT_FLIPADST
    { aom_iadst8_c, aom_iadst4_c },  // FLIPADST_FLIPADST
    { aom_iadst8_c, aom_iadst4_c },  // ADST_FLIPADST
    { aom_iadst8_c, aom_iadst4_c },  // FLIPADST_ADST
    { iidtx8_c, iidtx4_c },          // IDTX
    { aom_idct8_c, iidtx4_c },       // V_DCT
    { iidtx8_c, aom_idct4_c },       // H_DCT
    { aom_iadst8_c, iidtx4_c },      // V_ADST
    { iidtx8_c, aom_iadst4_c },      // H_ADST
    { aom_iadst8_c, iidtx4_c },      // V_FLIPADST
    { iidtx8_c, aom_iadst4_c },      // H_FLIPADST
#endif
#endif
  };

  const int n = 4;
  const int n2 = 8;
  int i, j;
  tran_low_t out[4][8], tmp[4][8], outtmp[4];
  tran_low_t *outp = &out[0][0];
  int outstride = n2;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_col = get_lgt8(txfm_param, 1, lgtmtx_col);
  int use_lgt_row = get_lgt4(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n2; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt4(input, outtmp, lgtmtx_row[0]);
    else
#endif
      IHT_4x8[tx_type].rows(input, outtmp);
    for (j = 0; j < n; ++j) {
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT8
      tmp[j][i] = outtmp[j];
#else
      tmp[j][i] = (tran_low_t)dct_const_round_shift(outtmp[j] * Sqrt2);
#endif
    }
    input += n;
  }

  // inverse transform column vectors
  for (i = 0; i < n; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt8(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_4x8[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n2, n);
#endif

  // Sum with the destination
  for (i = 0; i < n2; ++i) {
    for (j = 0; j < n; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT8
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 5));
#endif
    }
  }
}

void av1_iht8x4_32_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                         const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_8x4[] = {
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT8
    { daala_idct4, daala_idct8 },  // DCT_DCT  = 0
    { daala_idst4, daala_idct8 },  // ADST_DCT = 1
    { daala_idct4, daala_idst8 },  // DCT_ADST = 2
    { daala_idst4, daala_idst8 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst4, daala_idct8 },  // FLIPADST_DCT
    { daala_idct4, daala_idst8 },  // DCT_FLIPADST
    { daala_idst4, daala_idst8 },  // FLIPADST_FLIPADST
    { daala_idst4, daala_idst8 },  // ADST_FLIPADST
    { daala_idst4, daala_idst8 },  // FLIPADST_ADST
    { daala_idtx4, daala_idtx8 },  // IDTX
    { daala_idct4, daala_idtx8 },  // V_DCT
    { daala_idtx4, daala_idct8 },  // H_DCT
    { daala_idst4, daala_idtx8 },  // V_ADST
    { daala_idtx4, daala_idst8 },  // H_ADST
    { daala_idst4, daala_idtx8 },  // V_FLIPADST
    { daala_idtx4, daala_idst8 },  // H_FLIPADST
#endif
#else
    { aom_idct4_c, aom_idct8_c },    // DCT_DCT
    { aom_iadst4_c, aom_idct8_c },   // ADST_DCT
    { aom_idct4_c, aom_iadst8_c },   // DCT_ADST
    { aom_iadst4_c, aom_iadst8_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst4_c, aom_idct8_c },   // FLIPADST_DCT
    { aom_idct4_c, aom_iadst8_c },   // DCT_FLIPADST
    { aom_iadst4_c, aom_iadst8_c },  // FLIPADST_FLIPADST
    { aom_iadst4_c, aom_iadst8_c },  // ADST_FLIPADST
    { aom_iadst4_c, aom_iadst8_c },  // FLIPADST_ADST
    { iidtx4_c, iidtx8_c },          // IDTX
    { aom_idct4_c, iidtx8_c },       // V_DCT
    { iidtx4_c, aom_idct8_c },       // H_DCT
    { aom_iadst4_c, iidtx8_c },      // V_ADST
    { iidtx4_c, aom_iadst8_c },      // H_ADST
    { aom_iadst4_c, iidtx8_c },      // V_FLIPADST
    { iidtx4_c, aom_iadst8_c },      // H_FLIPADST
#endif
#endif
  };

  const int n = 4;
  const int n2 = 8;

  int i, j;
  tran_low_t out[8][4], tmp[8][4], outtmp[8];
  tran_low_t *outp = &out[0][0];
  int outstride = n;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_col = get_lgt4(txfm_param, 1, lgtmtx_col);
  int use_lgt_row = get_lgt8(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt8(input, outtmp, lgtmtx_row[0]);
    else
#endif
      IHT_8x4[tx_type].rows(input, outtmp);
    for (j = 0; j < n2; ++j) {
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT8
      tmp[j][i] = outtmp[j];
#else
      tmp[j][i] = (tran_low_t)dct_const_round_shift(outtmp[j] * Sqrt2);
#endif
    }
    input += n2;
  }

  // inverse transform column vectors
  for (i = 0; i < n2; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt4(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_8x4[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n, n2);
#endif

  // Sum with the destination
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n2; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT8
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 5));
#endif
    }
  }
}

void av1_iht4x16_64_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                          const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_4x16[] = {
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT16
    { daala_idct16, daala_idct4 },  // DCT_DCT  = 0
    { daala_idst16, daala_idct4 },  // ADST_DCT = 1
    { daala_idct16, daala_idst4 },  // DCT_ADST = 2
    { daala_idst16, daala_idst4 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst16, daala_idct4 },  // FLIPADST_DCT
    { daala_idct16, daala_idst4 },  // DCT_FLIPADST
    { daala_idst16, daala_idst4 },  // FLIPADST_FLIPADST
    { daala_idst16, daala_idst4 },  // ADST_FLIPADST
    { daala_idst16, daala_idst4 },  // FLIPADST_ADST
    { daala_idtx16, daala_idtx4 },  // IDTX
    { daala_idct16, daala_idtx4 },  // V_DCT
    { daala_idtx16, daala_idct4 },  // H_DCT
    { daala_idst16, daala_idtx4 },  // V_ADST
    { daala_idtx16, daala_idst4 },  // H_ADST
    { daala_idst16, daala_idtx4 },  // V_FLIPADST
    { daala_idtx16, daala_idst4 },  // H_FLIPADST
#endif
#else
    { aom_idct16_c, aom_idct4_c },    // DCT_DCT
    { aom_iadst16_c, aom_idct4_c },   // ADST_DCT
    { aom_idct16_c, aom_iadst4_c },   // DCT_ADST
    { aom_iadst16_c, aom_iadst4_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst16_c, aom_idct4_c },   // FLIPADST_DCT
    { aom_idct16_c, aom_iadst4_c },   // DCT_FLIPADST
    { aom_iadst16_c, aom_iadst4_c },  // FLIPADST_FLIPADST
    { aom_iadst16_c, aom_iadst4_c },  // ADST_FLIPADST
    { aom_iadst16_c, aom_iadst4_c },  // FLIPADST_ADST
    { iidtx16_c, iidtx4_c },          // IDTX
    { aom_idct16_c, iidtx4_c },       // V_DCT
    { iidtx16_c, aom_idct4_c },       // H_DCT
    { aom_iadst16_c, iidtx4_c },      // V_ADST
    { iidtx16_c, aom_iadst4_c },      // H_ADST
    { aom_iadst16_c, iidtx4_c },      // V_FLIPADST
    { iidtx16_c, aom_iadst4_c },      // H_FLIPADST
#endif
#endif
  };

  const int n = 4;
  const int n4 = 16;
  int i, j;
  tran_low_t out[4][16], tmp[4][16], outtmp[4];
  tran_low_t *outp = &out[0][0];
  int outstride = n4;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_row = get_lgt4(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n4; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt4(input, outtmp, lgtmtx_row[0]);
    else
#endif
      IHT_4x16[tx_type].rows(input, outtmp);
    for (j = 0; j < n; ++j) tmp[j][i] = outtmp[j];
    input += n;
  }

  // inverse transform column vectors
  for (i = 0; i < n; ++i) {
    IHT_4x16[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n4, n);
#endif

  // Sum with the destination
  for (i = 0; i < n4; ++i) {
    for (j = 0; j < n; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT16
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 5));
#endif
    }
  }
}

void av1_iht16x4_64_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                          const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_16x4[] = {
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT16
    { daala_idct4, daala_idct16 },  // DCT_DCT  = 0
    { daala_idst4, daala_idct16 },  // ADST_DCT = 1
    { daala_idct4, daala_idst16 },  // DCT_ADST = 2
    { daala_idst4, daala_idst16 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst4, daala_idct16 },  // FLIPADST_DCT
    { daala_idct4, daala_idst16 },  // DCT_FLIPADST
    { daala_idst4, daala_idst16 },  // FLIPADST_FLIPADST
    { daala_idst4, daala_idst16 },  // ADST_FLIPADST
    { daala_idst4, daala_idst16 },  // FLIPADST_ADST
    { daala_idtx4, daala_idtx16 },  // IDTX
    { daala_idct4, daala_idtx16 },  // V_DCT
    { daala_idtx4, daala_idct16 },  // H_DCT
    { daala_idst4, daala_idtx16 },  // V_ADST
    { daala_idtx4, daala_idst16 },  // H_ADST
    { daala_idst4, daala_idtx16 },  // V_FLIPADST
    { daala_idtx4, daala_idst16 },  // H_FLIPADST
#endif
#else
    { aom_idct4_c, aom_idct16_c },    // DCT_DCT
    { aom_iadst4_c, aom_idct16_c },   // ADST_DCT
    { aom_idct4_c, aom_iadst16_c },   // DCT_ADST
    { aom_iadst4_c, aom_iadst16_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst4_c, aom_idct16_c },   // FLIPADST_DCT
    { aom_idct4_c, aom_iadst16_c },   // DCT_FLIPADST
    { aom_iadst4_c, aom_iadst16_c },  // FLIPADST_FLIPADST
    { aom_iadst4_c, aom_iadst16_c },  // ADST_FLIPADST
    { aom_iadst4_c, aom_iadst16_c },  // FLIPADST_ADST
    { iidtx4_c, iidtx16_c },          // IDTX
    { aom_idct4_c, iidtx16_c },       // V_DCT
    { iidtx4_c, aom_idct16_c },       // H_DCT
    { aom_iadst4_c, iidtx16_c },      // V_ADST
    { iidtx4_c, aom_iadst16_c },      // H_ADST
    { aom_iadst4_c, iidtx16_c },      // V_FLIPADST
    { iidtx4_c, aom_iadst16_c },      // H_FLIPADST
#endif
#endif
  };

  const int n = 4;
  const int n4 = 16;

  int i, j;
  tran_low_t out[16][4], tmp[16][4], outtmp[16];
  tran_low_t *outp = &out[0][0];
  int outstride = n;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  int use_lgt_col = get_lgt4(txfm_param, 1, lgtmtx_col);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n; ++i) {
    IHT_16x4[tx_type].rows(input, outtmp);
    for (j = 0; j < n4; ++j) tmp[j][i] = outtmp[j];
    input += n4;
  }

  // inverse transform column vectors
  for (i = 0; i < n4; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt4(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_16x4[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n, n4);
#endif

  // Sum with the destination
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n4; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT4 && CONFIG_DAALA_DCT16
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 5));
#endif
    }
  }
}

void av1_iht8x16_128_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                           const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_8x16[] = {
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT16
    { daala_idct16, daala_idct8 },  // DCT_DCT  = 0
    { daala_idst16, daala_idct8 },  // ADST_DCT = 1
    { daala_idct16, daala_idst8 },  // DCT_ADST = 2
    { daala_idst16, daala_idst8 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst16, daala_idct8 },  // FLIPADST_DCT
    { daala_idct16, daala_idst8 },  // DCT_FLIPADST
    { daala_idst16, daala_idst8 },  // FLIPADST_FLIPADST
    { daala_idst16, daala_idst8 },  // ADST_FLIPADST
    { daala_idst16, daala_idst8 },  // FLIPADST_ADST
    { daala_idtx16, daala_idtx8 },  // IDTX
    { daala_idct16, daala_idtx8 },  // V_DCT
    { daala_idtx16, daala_idct8 },  // H_DCT
    { daala_idst16, daala_idtx8 },  // V_ADST
    { daala_idtx16, daala_idst8 },  // H_ADST
    { daala_idst16, daala_idtx8 },  // V_FLIPADST
    { daala_idtx16, daala_idst8 },  // H_FLIPADST
#endif
#else
    { aom_idct16_c, aom_idct8_c },    // DCT_DCT
    { aom_iadst16_c, aom_idct8_c },   // ADST_DCT
    { aom_idct16_c, aom_iadst8_c },   // DCT_ADST
    { aom_iadst16_c, aom_iadst8_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst16_c, aom_idct8_c },   // FLIPADST_DCT
    { aom_idct16_c, aom_iadst8_c },   // DCT_FLIPADST
    { aom_iadst16_c, aom_iadst8_c },  // FLIPADST_FLIPADST
    { aom_iadst16_c, aom_iadst8_c },  // ADST_FLIPADST
    { aom_iadst16_c, aom_iadst8_c },  // FLIPADST_ADST
    { iidtx16_c, iidtx8_c },          // IDTX
    { aom_idct16_c, iidtx8_c },       // V_DCT
    { iidtx16_c, aom_idct8_c },       // H_DCT
    { aom_iadst16_c, iidtx8_c },      // V_ADST
    { iidtx16_c, aom_iadst8_c },      // H_ADST
    { aom_iadst16_c, iidtx8_c },      // V_FLIPADST
    { iidtx16_c, aom_iadst8_c },      // H_FLIPADST
#endif
#endif
  };
  const int n = 8;
  const int n2 = 16;
  int i, j;
  tran_low_t out[8][16], tmp[8][16], outtmp[8];
  tran_low_t *outp = &out[0][0];
  int outstride = n2;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_row = get_lgt8(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n2; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt8(input, outtmp, lgtmtx_row[0]);
    else
#endif
      IHT_8x16[tx_type].rows(input, outtmp);
    for (j = 0; j < n; ++j) {
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT16
      tmp[j][i] = outtmp[j];
#else
      tmp[j][i] = (tran_low_t)dct_const_round_shift(outtmp[j] * Sqrt2);
#endif
    }
    input += n;
  }

  // inverse transform column vectors
  for (i = 0; i < n; ++i) {
    IHT_8x16[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n2, n);
#endif

  // Sum with the destination
  for (i = 0; i < n2; ++i) {
    for (j = 0; j < n; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT16
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

void av1_iht16x8_128_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                           const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_16x8[] = {
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT16
    { daala_idct8, daala_idct16 },  // DCT_DCT  = 0
    { daala_idst8, daala_idct16 },  // ADST_DCT = 1
    { daala_idct8, daala_idst16 },  // DCT_ADST = 2
    { daala_idst8, daala_idst16 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst8, daala_idct16 },  // FLIPADST_DCT
    { daala_idct8, daala_idst16 },  // DCT_FLIPADST
    { daala_idst8, daala_idst16 },  // FLIPADST_FLIPADST
    { daala_idst8, daala_idst16 },  // ADST_FLIPADST
    { daala_idst8, daala_idst16 },  // FLIPADST_ADST
    { daala_idtx8, daala_idtx16 },  // IDTX
    { daala_idct8, daala_idtx16 },  // V_DCT
    { daala_idtx8, daala_idct16 },  // H_DCT
    { daala_idst8, daala_idtx16 },  // V_ADST
    { daala_idtx8, daala_idst16 },  // H_ADST
    { daala_idst8, daala_idtx16 },  // V_FLIPADST
    { daala_idtx8, daala_idst16 },  // H_FLIPADST
#endif
#else
    { aom_idct8_c, aom_idct16_c },    // DCT_DCT
    { aom_iadst8_c, aom_idct16_c },   // ADST_DCT
    { aom_idct8_c, aom_iadst16_c },   // DCT_ADST
    { aom_iadst8_c, aom_iadst16_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst8_c, aom_idct16_c },   // FLIPADST_DCT
    { aom_idct8_c, aom_iadst16_c },   // DCT_FLIPADST
    { aom_iadst8_c, aom_iadst16_c },  // FLIPADST_FLIPADST
    { aom_iadst8_c, aom_iadst16_c },  // ADST_FLIPADST
    { aom_iadst8_c, aom_iadst16_c },  // FLIPADST_ADST
    { iidtx8_c, iidtx16_c },          // IDTX
    { aom_idct8_c, iidtx16_c },       // V_DCT
    { iidtx8_c, aom_idct16_c },       // H_DCT
    { aom_iadst8_c, iidtx16_c },      // V_ADST
    { iidtx8_c, aom_iadst16_c },      // H_ADST
    { aom_iadst8_c, iidtx16_c },      // V_FLIPADST
    { iidtx8_c, aom_iadst16_c },      // H_FLIPADST
#endif
#endif
  };

  const int n = 8;
  const int n2 = 16;

  int i, j;
  tran_low_t out[16][8], tmp[16][8], outtmp[16];
  tran_low_t *outp = &out[0][0];
  int outstride = n;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  int use_lgt_col = get_lgt8(txfm_param, 1, lgtmtx_col);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n; ++i) {
    IHT_16x8[tx_type].rows(input, outtmp);
    for (j = 0; j < n2; ++j) {
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT16
      tmp[j][i] = outtmp[j];
#else
      tmp[j][i] = (tran_low_t)dct_const_round_shift(outtmp[j] * Sqrt2);
#endif
    }
    input += n2;
  }

  // inverse transform column vectors
  for (i = 0; i < n2; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt8(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_16x8[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n, n2);
#endif

  // Sum with the destination
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n2; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT16
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

void av1_iht8x32_256_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                           const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_8x32[] = {
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT32
    { daala_idct32, daala_idct8 },  // DCT_DCT  = 0
    { daala_idst32, daala_idct8 },  // ADST_DCT = 1
    { daala_idct32, daala_idst8 },  // DCT_ADST = 2
    { daala_idst32, daala_idst8 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst32, daala_idct8 },  // FLIPADST_DCT
    { daala_idct32, daala_idst8 },  // DCT_FLIPADST
    { daala_idst32, daala_idst8 },  // FLIPADST_FLIPADST
    { daala_idst32, daala_idst8 },  // ADST_FLIPADST
    { daala_idst32, daala_idst8 },  // FLIPADST_ADST
    { daala_idtx32, daala_idtx8 },  // IDTX
    { daala_idct32, daala_idtx8 },  // V_DCT
    { daala_idtx32, daala_idct8 },  // H_DCT
    { daala_idst32, daala_idtx8 },  // V_ADST
    { daala_idtx32, daala_idst8 },  // H_ADST
    { daala_idst32, daala_idtx8 },  // V_FLIPADST
    { daala_idtx32, daala_idst8 },  // H_FLIPADST
#endif
#else
    { aom_idct32_c, aom_idct8_c },     // DCT_DCT
    { ihalfright32_c, aom_idct8_c },   // ADST_DCT
    { aom_idct32_c, aom_iadst8_c },    // DCT_ADST
    { ihalfright32_c, aom_iadst8_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { ihalfright32_c, aom_idct8_c },   // FLIPADST_DCT
    { aom_idct32_c, aom_iadst8_c },    // DCT_FLIPADST
    { ihalfright32_c, aom_iadst8_c },  // FLIPADST_FLIPADST
    { ihalfright32_c, aom_iadst8_c },  // ADST_FLIPADST
    { ihalfright32_c, aom_iadst8_c },  // FLIPADST_ADST
    { iidtx32_c, iidtx8_c },           // IDTX
    { aom_idct32_c, iidtx8_c },        // V_DCT
    { iidtx32_c, aom_idct8_c },        // H_DCT
    { ihalfright32_c, iidtx8_c },      // V_ADST
    { iidtx32_c, aom_iadst8_c },       // H_ADST
    { ihalfright32_c, iidtx8_c },      // V_FLIPADST
    { iidtx32_c, aom_iadst8_c },       // H_FLIPADST
#endif
#endif
  };

  const int n = 8;
  const int n4 = 32;
  int i, j;
  tran_low_t out[8][32], tmp[8][32], outtmp[8];
  tran_low_t *outp = &out[0][0];
  int outstride = n4;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_row = get_lgt8(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n4; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt8(input, outtmp, lgtmtx_row[0]);
    else
#endif
      IHT_8x32[tx_type].rows(input, outtmp);
    for (j = 0; j < n; ++j) tmp[j][i] = outtmp[j];
    input += n;
  }

  // inverse transform column vectors
  for (i = 0; i < n; ++i) {
    IHT_8x32[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n4, n);
#endif

  // Sum with the destination
  for (i = 0; i < n4; ++i) {
    for (j = 0; j < n; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT32
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

void av1_iht32x8_256_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                           const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_32x8[] = {
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT32
    { daala_idct8, daala_idct32 },  // DCT_DCT  = 0
    { daala_idst8, daala_idct32 },  // ADST_DCT = 1
    { daala_idct8, daala_idst32 },  // DCT_ADST = 2
    { daala_idst8, daala_idst32 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst8, daala_idct32 },  // FLIPADST_DCT
    { daala_idct8, daala_idst32 },  // DCT_FLIPADST
    { daala_idst8, daala_idst32 },  // FLIPADST_FLIPADST
    { daala_idst8, daala_idst32 },  // ADST_FLIPADST
    { daala_idst8, daala_idst32 },  // FLIPADST_ADST
    { daala_idtx8, daala_idtx32 },  // IDTX
    { daala_idct8, daala_idtx32 },  // V_DCT
    { daala_idtx8, daala_idct32 },  // H_DCT
    { daala_idst8, daala_idtx32 },  // V_ADST
    { daala_idtx8, daala_idst32 },  // H_ADST
    { daala_idst8, daala_idtx32 },  // V_FLIPADST
    { daala_idtx8, daala_idst32 },  // H_FLIPADST
#endif
#else
    { aom_idct8_c, aom_idct32_c },     // DCT_DCT
    { aom_iadst8_c, aom_idct32_c },    // ADST_DCT
    { aom_idct8_c, ihalfright32_c },   // DCT_ADST
    { aom_iadst8_c, ihalfright32_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst8_c, aom_idct32_c },    // FLIPADST_DCT
    { aom_idct8_c, ihalfright32_c },   // DCT_FLIPADST
    { aom_iadst8_c, ihalfright32_c },  // FLIPADST_FLIPADST
    { aom_iadst8_c, ihalfright32_c },  // ADST_FLIPADST
    { aom_iadst8_c, ihalfright32_c },  // FLIPADST_ADST
    { iidtx8_c, iidtx32_c },           // IDTX
    { aom_idct8_c, iidtx32_c },        // V_DCT
    { iidtx8_c, aom_idct32_c },        // H_DCT
    { aom_iadst8_c, iidtx32_c },       // V_ADST
    { iidtx8_c, ihalfright32_c },      // H_ADST
    { aom_iadst8_c, iidtx32_c },       // V_FLIPADST
    { iidtx8_c, ihalfright32_c },      // H_FLIPADST
#endif
#endif
  };

  const int n = 8;
  const int n4 = 32;

  int i, j;
  tran_low_t out[32][8], tmp[32][8], outtmp[32];
  tran_low_t *outp = &out[0][0];
  int outstride = n;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  int use_lgt_col = get_lgt4(txfm_param, 1, lgtmtx_col);
#endif

  // inverse transform row vectors and transpose
  for (i = 0; i < n; ++i) {
    IHT_32x8[tx_type].rows(input, outtmp);
    for (j = 0; j < n4; ++j) tmp[j][i] = outtmp[j];
    input += n4;
  }

  // inverse transform column vectors
  for (i = 0; i < n4; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt8(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_32x8[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n, n4);
#endif

  // Sum with the destination
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n4; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT8 && CONFIG_DAALA_DCT32
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

void av1_iht16x32_512_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                            const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_16x32[] = {
#if CONFIG_DAALA_DCT16 && CONFIG_DAALA_DCT32
    { daala_idct32, daala_idct16 },  // DCT_DCT  = 0
    { daala_idst32, daala_idct16 },  // ADST_DCT = 1
    { daala_idct32, daala_idst16 },  // DCT_ADST = 2
    { daala_idst32, daala_idst16 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst32, daala_idct16 },  // FLIPADST_DCT
    { daala_idct32, daala_idst16 },  // DCT_FLIPADST
    { daala_idst32, daala_idst16 },  // FLIPADST_FLIPADST
    { daala_idst32, daala_idst16 },  // ADST_FLIPADST
    { daala_idst32, daala_idst16 },  // FLIPADST_ADST
    { daala_idtx32, daala_idtx16 },  // IDTX
    { daala_idct32, daala_idtx16 },  // V_DCT
    { daala_idtx32, daala_idct16 },  // H_DCT
    { daala_idst32, daala_idtx16 },  // V_ADST
    { daala_idtx32, daala_idst16 },  // H_ADST
    { daala_idst32, daala_idtx16 },  // V_FLIPADST
    { daala_idtx32, daala_idst16 },  // H_FLIPADST
#endif
#else
    { aom_idct32_c, aom_idct16_c },     // DCT_DCT
    { ihalfright32_c, aom_idct16_c },   // ADST_DCT
    { aom_idct32_c, aom_iadst16_c },    // DCT_ADST
    { ihalfright32_c, aom_iadst16_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { ihalfright32_c, aom_idct16_c },   // FLIPADST_DCT
    { aom_idct32_c, aom_iadst16_c },    // DCT_FLIPADST
    { ihalfright32_c, aom_iadst16_c },  // FLIPADST_FLIPADST
    { ihalfright32_c, aom_iadst16_c },  // ADST_FLIPADST
    { ihalfright32_c, aom_iadst16_c },  // FLIPADST_ADST
    { iidtx32_c, iidtx16_c },           // IDTX
    { aom_idct32_c, iidtx16_c },        // V_DCT
    { iidtx32_c, aom_idct16_c },        // H_DCT
    { ihalfright32_c, iidtx16_c },      // V_ADST
    { iidtx32_c, aom_iadst16_c },       // H_ADST
    { ihalfright32_c, iidtx16_c },      // V_FLIPADST
    { iidtx32_c, aom_iadst16_c },       // H_FLIPADST
#endif
#endif
  };

  const int n = 16;
  const int n2 = 32;
  int i, j;
  tran_low_t out[16][32], tmp[16][32], outtmp[16];
  tran_low_t *outp = &out[0][0];
  int outstride = n2;

  // inverse transform row vectors and transpose
  for (i = 0; i < n2; ++i) {
    IHT_16x32[tx_type].rows(input, outtmp);
    for (j = 0; j < n; ++j) {
#if CONFIG_DAALA_DCT16 && CONFIG_DAALA_DCT32
      tmp[j][i] = outtmp[j];
#else
      tmp[j][i] = (tran_low_t)dct_const_round_shift(outtmp[j] * Sqrt2);
#endif
    }
    input += n;
  }

  // inverse transform column vectors
  for (i = 0; i < n; ++i) IHT_16x32[tx_type].cols(tmp[i], out[i]);

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n2, n);
#endif

  // Sum with the destination
  for (i = 0; i < n2; ++i) {
    for (j = 0; j < n; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT16 && CONFIG_DAALA_DCT32
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

void av1_iht32x16_512_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                            const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_32x16[] = {
#if CONFIG_DAALA_DCT16 && CONFIG_DAALA_DCT32
    { daala_idct16, daala_idct32 },  // DCT_DCT  = 0
    { daala_idst16, daala_idct32 },  // ADST_DCT = 1
    { daala_idct16, daala_idst32 },  // DCT_ADST = 2
    { daala_idst16, daala_idst32 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst16, daala_idct32 },  // FLIPADST_DCT
    { daala_idct16, daala_idst32 },  // DCT_FLIPADST
    { daala_idst16, daala_idst32 },  // FLIPADST_FLIPADST
    { daala_idst16, daala_idst32 },  // ADST_FLIPADST
    { daala_idst16, daala_idst32 },  // FLIPADST_ADST
    { daala_idtx16, daala_idtx32 },  // IDTX
    { daala_idct16, daala_idtx32 },  // V_DCT
    { daala_idtx16, daala_idct32 },  // H_DCT
    { daala_idst16, daala_idtx32 },  // V_ADST
    { daala_idtx16, daala_idst32 },  // H_ADST
    { daala_idst16, daala_idtx32 },  // V_FLIPADST
    { daala_idtx16, daala_idst32 },  // H_FLIPADST
#endif
#else
    { aom_idct16_c, aom_idct32_c },     // DCT_DCT
    { aom_iadst16_c, aom_idct32_c },    // ADST_DCT
    { aom_idct16_c, ihalfright32_c },   // DCT_ADST
    { aom_iadst16_c, ihalfright32_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { aom_iadst16_c, aom_idct32_c },    // FLIPADST_DCT
    { aom_idct16_c, ihalfright32_c },   // DCT_FLIPADST
    { aom_iadst16_c, ihalfright32_c },  // FLIPADST_FLIPADST
    { aom_iadst16_c, ihalfright32_c },  // ADST_FLIPADST
    { aom_iadst16_c, ihalfright32_c },  // FLIPADST_ADST
    { iidtx16_c, iidtx32_c },           // IDTX
    { aom_idct16_c, iidtx32_c },        // V_DCT
    { iidtx16_c, aom_idct32_c },        // H_DCT
    { aom_iadst16_c, iidtx32_c },       // V_ADST
    { iidtx16_c, ihalfright32_c },      // H_ADST
    { aom_iadst16_c, iidtx32_c },       // V_FLIPADST
    { iidtx16_c, ihalfright32_c },      // H_FLIPADST
#endif
#endif
  };
  const int n = 16;
  const int n2 = 32;

  int i, j;
  tran_low_t out[32][16], tmp[32][16], outtmp[32];
  tran_low_t *outp = &out[0][0];
  int outstride = n;

  // inverse transform row vectors and transpose
  for (i = 0; i < n; ++i) {
    IHT_32x16[tx_type].rows(input, outtmp);
    for (j = 0; j < n2; ++j) {
#if CONFIG_DAALA_DCT16 && CONFIG_DAALA_DCT32
      tmp[j][i] = outtmp[j];
#else
      tmp[j][i] = (tran_low_t)dct_const_round_shift(outtmp[j] * Sqrt2);
#endif
    }
    input += n2;
  }

  // inverse transform column vectors
  for (i = 0; i < n2; ++i) IHT_32x16[tx_type].cols(tmp[i], out[i]);

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, n, n2);
#endif

  // Sum with the destination
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n2; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT16 && CONFIG_DAALA_DCT32
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

void av1_iht8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                         const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_8[] = {
#if CONFIG_DAALA_DCT8
    { daala_idct8, daala_idct8 },  // DCT_DCT  = 0
    { daala_idst8, daala_idct8 },  // ADST_DCT = 1
    { daala_idct8, daala_idst8 },  // DCT_ADST = 2
    { daala_idst8, daala_idst8 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst8, daala_idct8 },  // FLIPADST_DCT
    { daala_idct8, daala_idst8 },  // DCT_FLIPADST
    { daala_idst8, daala_idst8 },  // FLIPADST_FLIPADST
    { daala_idst8, daala_idst8 },  // ADST_FLIPADST
    { daala_idst8, daala_idst8 },  // FLIPADST_ADST
    { daala_idtx8, daala_idtx8 },  // IDTX
    { daala_idct8, daala_idtx8 },  // V_DCT
    { daala_idtx8, daala_idct8 },  // H_DCT
    { daala_idst8, daala_idtx8 },  // V_ADST
    { daala_idtx8, daala_idst8 },  // H_ADST
    { daala_idst8, daala_idtx8 },  // V_FLIPADST
    { daala_idtx8, daala_idst8 },  // H_FLIPADST
#endif
#else
    { aom_idct8_c, aom_idct8_c },    // DCT_DCT  = 0
    { aom_iadst8_c, aom_idct8_c },   // ADST_DCT = 1
    { aom_idct8_c, aom_iadst8_c },   // DCT_ADST = 2
    { aom_iadst8_c, aom_iadst8_c },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { aom_iadst8_c, aom_idct8_c },   // FLIPADST_DCT
    { aom_idct8_c, aom_iadst8_c },   // DCT_FLIPADST
    { aom_iadst8_c, aom_iadst8_c },  // FLIPADST_FLIPADST
    { aom_iadst8_c, aom_iadst8_c },  // ADST_FLIPADST
    { aom_iadst8_c, aom_iadst8_c },  // FLIPADST_ADST
    { iidtx8_c, iidtx8_c },          // IDTX
    { aom_idct8_c, iidtx8_c },       // V_DCT
    { iidtx8_c, aom_idct8_c },       // H_DCT
    { aom_iadst8_c, iidtx8_c },      // V_ADST
    { iidtx8_c, aom_iadst8_c },      // H_ADST
    { aom_iadst8_c, iidtx8_c },      // V_FLIPADST
    { iidtx8_c, aom_iadst8_c },      // H_FLIPADST
#endif
#endif
  };

  int i, j;
  tran_low_t tmp[8][8];
  tran_low_t out[8][8];
  tran_low_t *outp = &out[0][0];
  int outstride = 8;

#if CONFIG_LGT
  const tran_high_t *lgtmtx_col[1];
  const tran_high_t *lgtmtx_row[1];
  int use_lgt_col = get_lgt8(txfm_param, 1, lgtmtx_col);
  int use_lgt_row = get_lgt8(txfm_param, 0, lgtmtx_row);
#endif

  // inverse transform row vectors
  for (i = 0; i < 8; ++i) {
#if CONFIG_LGT
    if (use_lgt_row)
      ilgt8(input, out[i], lgtmtx_row[0]);
    else
#endif
      IHT_8[tx_type].rows(input, out[i]);
    input += 8;
  }

  // transpose
  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++) {
      tmp[j][i] = out[i][j];
    }
  }

  // inverse transform column vectors
  for (i = 0; i < 8; ++i) {
#if CONFIG_LGT
    if (use_lgt_col)
      ilgt8(tmp[i], out[i], lgtmtx_col[0]);
    else
#endif
      IHT_8[tx_type].cols(tmp[i], out[i]);
  }

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, 8, 8);
#endif

  // Sum with the destination
  for (i = 0; i < 8; ++i) {
    for (j = 0; j < 8; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT8
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 5));
#endif
    }
  }
}

void av1_iht16x16_256_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                            const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_16[] = {
#if CONFIG_DAALA_DCT16
    { daala_idct16, daala_idct16 },  // DCT_DCT  = 0
    { daala_idst16, daala_idct16 },  // ADST_DCT = 1
    { daala_idct16, daala_idst16 },  // DCT_ADST = 2
    { daala_idst16, daala_idst16 },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { daala_idst16, daala_idct16 },  // FLIPADST_DCT
    { daala_idct16, daala_idst16 },  // DCT_FLIPADST
    { daala_idst16, daala_idst16 },  // FLIPADST_FLIPADST
    { daala_idst16, daala_idst16 },  // ADST_FLIPADST
    { daala_idst16, daala_idst16 },  // FLIPADST_ADST
    { daala_idtx16, daala_idtx16 },  // IDTX
    { daala_idct16, daala_idtx16 },  // V_DCT
    { daala_idtx16, daala_idct16 },  // H_DCT
    { daala_idst16, daala_idtx16 },  // V_ADST
    { daala_idtx16, daala_idst16 },  // H_ADST
    { daala_idst16, daala_idtx16 },  // V_FLIPADST
    { daala_idtx16, daala_idst16 },  // H_FLIPADST
#endif
#else
    { aom_idct16_c, aom_idct16_c },    // DCT_DCT  = 0
    { aom_iadst16_c, aom_idct16_c },   // ADST_DCT = 1
    { aom_idct16_c, aom_iadst16_c },   // DCT_ADST = 2
    { aom_iadst16_c, aom_iadst16_c },  // ADST_ADST = 3
#if CONFIG_EXT_TX
    { aom_iadst16_c, aom_idct16_c },   // FLIPADST_DCT
    { aom_idct16_c, aom_iadst16_c },   // DCT_FLIPADST
    { aom_iadst16_c, aom_iadst16_c },  // FLIPADST_FLIPADST
    { aom_iadst16_c, aom_iadst16_c },  // ADST_FLIPADST
    { aom_iadst16_c, aom_iadst16_c },  // FLIPADST_ADST
    { iidtx16_c, iidtx16_c },          // IDTX
    { aom_idct16_c, iidtx16_c },       // V_DCT
    { iidtx16_c, aom_idct16_c },       // H_DCT
    { aom_iadst16_c, iidtx16_c },      // V_ADST
    { iidtx16_c, aom_iadst16_c },      // H_ADST
    { aom_iadst16_c, iidtx16_c },      // V_FLIPADST
    { iidtx16_c, aom_iadst16_c },      // H_FLIPADST
#endif
#endif
  };

  int i, j;
  tran_low_t tmp[16][16];
  tran_low_t out[16][16];
  tran_low_t *outp = &out[0][0];
  int outstride = 16;

  // inverse transform row vectors
  for (i = 0; i < 16; ++i) {
    IHT_16[tx_type].rows(input, out[i]);
    input += 16;
  }

  // transpose
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      tmp[j][i] = out[i][j];
    }
  }

  // inverse transform column vectors
  for (i = 0; i < 16; ++i) IHT_16[tx_type].cols(tmp[i], out[i]);

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, 16, 16);
#endif

  // Sum with the destination
  for (i = 0; i < 16; ++i) {
    for (j = 0; j < 16; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT16
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}

#if CONFIG_EXT_TX || CONFIG_DAALA_DCT32
void av1_iht32x32_1024_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_32[] = {
#if CONFIG_DAALA_DCT32
    { daala_idct32, daala_idct32 },  // DCT_DCT
#if CONFIG_EXT_TX
    { daala_idst32, daala_idct32 },  // ADST_DCT
    { daala_idct32, daala_idst32 },  // DCT_ADST
    { daala_idst32, daala_idst32 },  // ADST_ADST
    { daala_idst32, daala_idct32 },  // FLIPADST_DCT
    { daala_idct32, daala_idst32 },  // DCT_FLIPADST
    { daala_idst32, daala_idst32 },  // FLIPADST_FLIPADST
    { daala_idst32, daala_idst32 },  // ADST_FLIPADST
    { daala_idst32, daala_idst32 },  // FLIPADST_ADST
    { daala_idtx32, daala_idtx32 },  // IDTX
    { daala_idct32, daala_idtx32 },  // V_DCT
    { daala_idtx32, daala_idct32 },  // H_DCT
    { daala_idst32, daala_idtx32 },  // V_ADST
    { daala_idtx32, daala_idst32 },  // H_ADST
    { daala_idst32, daala_idtx32 },  // V_FLIPADST
    { daala_idtx32, daala_idst32 },  // H_FLIPADST
#endif
#else
    { aom_idct32_c, aom_idct32_c },      // DCT_DCT
#if CONFIG_EXT_TX
    { ihalfright32_c, aom_idct32_c },    // ADST_DCT
    { aom_idct32_c, ihalfright32_c },    // DCT_ADST
    { ihalfright32_c, ihalfright32_c },  // ADST_ADST
    { ihalfright32_c, aom_idct32_c },    // FLIPADST_DCT
    { aom_idct32_c, ihalfright32_c },    // DCT_FLIPADST
    { ihalfright32_c, ihalfright32_c },  // FLIPADST_FLIPADST
    { ihalfright32_c, ihalfright32_c },  // ADST_FLIPADST
    { ihalfright32_c, ihalfright32_c },  // FLIPADST_ADST
    { iidtx32_c, iidtx32_c },            // IDTX
    { aom_idct32_c, iidtx32_c },         // V_DCT
    { iidtx32_c, aom_idct32_c },         // H_DCT
    { ihalfright32_c, iidtx32_c },       // V_ADST
    { iidtx32_c, ihalfright32_c },       // H_ADST
    { ihalfright32_c, iidtx32_c },       // V_FLIPADST
    { iidtx32_c, ihalfright32_c },       // H_FLIPADST
#endif
#endif
  };

  int i, j;
  tran_low_t tmp[32][32];
  tran_low_t out[32][32];
  tran_low_t *outp = &out[0][0];
  int outstride = 32;

  // inverse transform row vectors
  for (i = 0; i < 32; ++i) {
    IHT_32[tx_type].rows(input, out[i]);
    input += 32;
  }

  // transpose
  for (i = 0; i < 32; i++) {
    for (j = 0; j < 32; j++) {
      tmp[j][i] = out[i][j];
    }
  }

  // inverse transform column vectors
  for (i = 0; i < 32; ++i) IHT_32[tx_type].cols(tmp[i], out[i]);

  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, 32, 32);

  // Sum with the destination
  for (i = 0; i < 32; ++i) {
    for (j = 0; j < 32; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT32
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 6));
#endif
    }
  }
}
#endif  // CONFIG_EXT_TX || CONFIG_DAALA_DCT32

#if CONFIG_TX64X64
void av1_iht64x64_4096_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  int tx_type = txfm_param->tx_type;
#if CONFIG_MRC_TX
  assert(tx_type != MRC_DCT && "Invalid tx type for tx size");
#endif  // CONFIG_MRC_TX
#if CONFIG_DCT_ONLY
  assert(tx_type == DCT_DCT);
#endif
  static const transform_2d IHT_64[] = {
#if CONFIG_DAALA_DCT64
    { daala_idct64, daala_idct64 },  // DCT_DCT
    { daala_idst64, daala_idct64 },  // ADST_DCT
    { daala_idct64, daala_idst64 },  // DCT_ADST
    { daala_idst64, daala_idst64 },  // ADST_ADST
#if CONFIG_EXT_TX
    { daala_idst64, daala_idct64 },  // FLIPADST_DCT
    { daala_idct64, daala_idst64 },  // DCT_FLIPADST
    { daala_idst64, daala_idst64 },  // FLIPADST_FLIPADST
    { daala_idst64, daala_idst64 },  // ADST_FLIPADST
    { daala_idst64, daala_idst64 },  // FLIPADST_ADST
    { daala_idtx64, daala_idtx64 },  // IDTX
    { daala_idct64, daala_idtx64 },  // V_DCT
    { daala_idtx64, daala_idct64 },  // H_DCT
    { daala_idst64, daala_idtx64 },  // V_ADST
    { daala_idtx64, daala_idst64 },  // H_ADST
    { daala_idst64, daala_idtx64 },  // V_FLIPADST
    { daala_idtx64, daala_idst64 },  // H_FLIPADST
#endif
#else
    { idct64_col_c, idct64_row_c },      // DCT_DCT
    { ihalfright64_c, idct64_row_c },    // ADST_DCT
    { idct64_col_c, ihalfright64_c },    // DCT_ADST
    { ihalfright64_c, ihalfright64_c },  // ADST_ADST
#if CONFIG_EXT_TX
    { ihalfright64_c, idct64_row_c },    // FLIPADST_DCT
    { idct64_col_c, ihalfright64_c },    // DCT_FLIPADST
    { ihalfright64_c, ihalfright64_c },  // FLIPADST_FLIPADST
    { ihalfright64_c, ihalfright64_c },  // ADST_FLIPADST
    { ihalfright64_c, ihalfright64_c },  // FLIPADST_ADST
    { iidtx64_c, iidtx64_c },            // IDTX
    { idct64_col_c, iidtx64_c },         // V_DCT
    { iidtx64_c, idct64_row_c },         // H_DCT
    { ihalfright64_c, iidtx64_c },       // V_ADST
    { iidtx64_c, ihalfright64_c },       // H_ADST
    { ihalfright64_c, iidtx64_c },       // V_FLIPADST
    { iidtx64_c, ihalfright64_c },       // H_FLIPADST
#endif
#endif
  };

  int i, j;
  tran_low_t tmp[64][64];
  tran_low_t out[64][64];
  tran_low_t *outp = &out[0][0];
  int outstride = 64;

  // inverse transform row vectors
  for (i = 0; i < 64; ++i) {
    IHT_64[tx_type].rows(input, out[i]);
#if !CONFIG_DAALA_DCT64
    for (j = 0; j < 64; ++j) out[i][j] = ROUND_POWER_OF_TWO(out[i][j], 1);
#endif
    input += 64;
  }

  // transpose
  for (i = 0; i < 64; i++) {
    for (j = 0; j < 64; j++) {
      tmp[j][i] = out[i][j];
    }
  }

  // inverse transform column vectors
  for (i = 0; i < 64; ++i) IHT_64[tx_type].cols(tmp[i], out[i]);

#if CONFIG_EXT_TX
  maybe_flip_strides(&dest, &stride, &outp, &outstride, tx_type, 64, 64);
#endif  // CONFIG_EXT_TX

  // Sum with the destination
  for (i = 0; i < 64; ++i) {
    for (j = 0; j < 64; ++j) {
      int d = i * stride + j;
      int s = j * outstride + i;
#if CONFIG_DAALA_DCT64
      dest[d] = clip_pixel_add(dest[d],
        ROUND_POWER_OF_TWO(outp[s], TX_COEFF_DEPTH - txfm_param->bd));
#else
      dest[d] = clip_pixel_add(dest[d], ROUND_POWER_OF_TWO(outp[s], 5));
#endif
    }
  }
}
#endif  // CONFIG_TX64X64

// idct
void av1_idct4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                     const TxfmParam *txfm_param) {
    av1_iht4x4_16_add_c(input, dest, stride, txfm_param);
}

void av1_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                     const TxfmParam *txfm_param) {
  const int eob = txfm_param->eob;
  if (eob > 1)
    aom_iwht4x4_16_add(input, dest, stride);
  else
    aom_iwht4x4_1_add(input, dest, stride);
}

#if !CONFIG_DAALA_DCT8
static void idct8x8_add(const tran_low_t *input, uint8_t *dest, int stride,
                        const TxfmParam *txfm_param) {
// If dc is 1, then input[0] is the reconstructed value, do not need
// dequantization. Also, when dc is 1, dc is counted in eobs, namely eobs >=1.

// The calculation can be simplified if there are not many non-zero dct
// coefficients. Use eobs to decide what to do.
// TODO(yunqingwang): "eobs = 1" case is also handled in av1_short_idct8x8_c.
// Combine that with code here.
#if CONFIG_ADAPT_SCAN
  const int16_t half = txfm_param->eob_threshold[0];
#else
  const int16_t half = 12;
#endif

  const int eob = txfm_param->eob;
  if (eob == 1)
    // DC only DCT coefficient
    aom_idct8x8_1_add(input, dest, stride);
  else if (eob <= half)
    aom_idct8x8_12_add(input, dest, stride);
  else
    aom_idct8x8_64_add(input, dest, stride);
}
#endif

#if !CONFIG_DAALA_DCT16
static void idct16x16_add(const tran_low_t *input, uint8_t *dest, int stride,
                          const TxfmParam *txfm_param) {
// The calculation can be simplified if there are not many non-zero dct
// coefficients. Use eobs to separate different cases.
#if CONFIG_ADAPT_SCAN
  const int16_t half = txfm_param->eob_threshold[0];
  const int16_t quarter = txfm_param->eob_threshold[1];
#else
  const int16_t half = 38;
  const int16_t quarter = 10;
#endif

  const int eob = txfm_param->eob;
  if (eob == 1) /* DC only DCT coefficient. */
    aom_idct16x16_1_add(input, dest, stride);
  else if (eob <= quarter)
    aom_idct16x16_10_add(input, dest, stride);
  else if (eob <= half)
    aom_idct16x16_38_add(input, dest, stride);
  else
    aom_idct16x16_256_add(input, dest, stride);
}
#endif

#if CONFIG_MRC_TX
static void imrc32x32_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                            const TxfmParam *txfm_param) {
#if CONFIG_ADAPT_SCAN
  const int16_t half = txfm_param->eob_threshold[0];
  const int16_t quarter = txfm_param->eob_threshold[1];
#else
  const int16_t half = 135;
  const int16_t quarter = 34;
#endif

  const int eob = txfm_param->eob;
  int n_masked_vals = 0;
  uint8_t *mask;
  uint8_t mask_tmp[32 * 32];
  if (eob == 1) {
    aom_idct32x32_1_add_c(input, dest, stride);
  } else {
    if ((txfm_param->is_inter && SIGNAL_MRC_MASK_INTER) ||
        (!txfm_param->is_inter && SIGNAL_MRC_MASK_INTRA)) {
      mask = txfm_param->mask;
    } else {
      n_masked_vals =
          get_mrc_pred_mask(txfm_param->dst, txfm_param->stride, mask_tmp, 32,
                            32, 32, txfm_param->is_inter);
      if (!is_valid_mrc_mask(n_masked_vals, 32, 32))
        assert(0 && "Invalid MRC mask");
      mask = mask_tmp;
    }
    if (eob <= quarter)
      // non-zero coeff only in upper-left 8x8
      aom_imrc32x32_34_add_c(input, dest, stride, mask);
    else if (eob <= half)
      // non-zero coeff only in upper-left 16x16
      aom_imrc32x32_135_add_c(input, dest, stride, mask);
    else
      aom_imrc32x32_1024_add_c(input, dest, stride, mask);
  }
}
#endif  // CONFIG_MRC_TX

#if !CONFIG_DAALA_DCT32
static void idct32x32_add(const tran_low_t *input, uint8_t *dest, int stride,
                          const TxfmParam *txfm_param) {
#if CONFIG_ADAPT_SCAN
  const int16_t half = txfm_param->eob_threshold[0];
  const int16_t quarter = txfm_param->eob_threshold[1];
#else
  const int16_t half = 135;
  const int16_t quarter = 34;
#endif

  const int eob = txfm_param->eob;
  if (eob == 1)
    aom_idct32x32_1_add(input, dest, stride);
  else if (eob <= quarter)
    // non-zero coeff only in upper-left 8x8
    aom_idct32x32_34_add(input, dest, stride);
  else if (eob <= half)
    // non-zero coeff only in upper-left 16x16
    aom_idct32x32_135_add(input, dest, stride);
  else
    aom_idct32x32_1024_add(input, dest, stride);
}
#endif

#if CONFIG_TX64X64 && !CONFIG_DAALA_DCT64
static void idct64x64_add(const tran_low_t *input, uint8_t *dest, int stride,
                          const TxfmParam *txfm_param) {
  (void)txfm_param;
  av1_iht64x64_4096_add(input, dest, stride, txfm_param);
}
#endif  // CONFIG_TX64X64 && !CONFIG_DAALA_DCT64

#if CONFIG_CHROMA_2X2
static void inv_txfm_add_2x2(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  tran_high_t a1 = input[0] >> 2;
  tran_high_t b1 = input[1] >> 2;
  tran_high_t c1 = input[2] >> 2;
  tran_high_t d1 = input[3] >> 2;

  tran_high_t a2 = a1 + c1;
  tran_high_t b2 = b1 + d1;
  tran_high_t c2 = a1 - c1;
  tran_high_t d2 = b1 - d1;

  (void)txfm_param;

  a1 = (a2 + b2) >> 2;
  b1 = (a2 - b2) >> 2;
  c1 = (c2 + d2) >> 2;
  d1 = (c2 - d2) >> 2;

  dest[0] = clip_pixel_add(dest[0], WRAPLOW(a1));
  dest[1] = clip_pixel_add(dest[1], WRAPLOW(b1));
  dest[stride] = clip_pixel_add(dest[stride], WRAPLOW(c1));
  dest[stride + 1] = clip_pixel_add(dest[stride + 1], WRAPLOW(d1));
}
#endif

static void inv_txfm_add_4x4(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  if (txfm_param->lossless) {
    assert(txfm_param->tx_type == DCT_DCT);
    av1_iwht4x4_add(input, dest, stride, txfm_param);
    return;
  }
  av1_iht4x4_16_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_4x8(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  av1_iht4x8_32_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_8x4(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  av1_iht8x4_32_add_c(input, dest, stride, txfm_param);
}

// These will be used by the masked-tx experiment in the future.
#if CONFIG_RECT_TX_EXT && (CONFIG_EXT_TX || CONFIG_VAR_TX)
static void inv_txfm_add_4x16(const tran_low_t *input, uint8_t *dest,
                              int stride, const TxfmParam *txfm_param) {
  av1_iht4x16_64_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_16x4(const tran_low_t *input, uint8_t *dest,
                              int stride, const TxfmParam *txfm_param) {
  av1_iht16x4_64_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_8x32(const tran_low_t *input, uint8_t *dest,
                              int stride, const TxfmParam *txfm_param) {
  av1_iht8x32_256_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_32x8(const tran_low_t *input, uint8_t *dest,
                              int stride, const TxfmParam *txfm_param) {
  av1_iht32x8_256_add_c(input, dest, stride, txfm_param);
}
#endif

static void inv_txfm_add_8x16(const tran_low_t *input, uint8_t *dest,
                              int stride, const TxfmParam *txfm_param) {
  av1_iht8x16_128_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_16x8(const tran_low_t *input, uint8_t *dest,
                              int stride, const TxfmParam *txfm_param) {
  av1_iht16x8_128_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_16x32(const tran_low_t *input, uint8_t *dest,
                               int stride, const TxfmParam *txfm_param) {
  av1_iht16x32_512_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_32x16(const tran_low_t *input, uint8_t *dest,
                               int stride, const TxfmParam *txfm_param) {
  av1_iht32x16_512_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_8x8(const tran_low_t *input, uint8_t *dest, int stride,
                             const TxfmParam *txfm_param) {
  av1_iht8x8_64_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_16x16(const tran_low_t *input, uint8_t *dest,
                               int stride, const TxfmParam *txfm_param) {
  av1_iht16x16_256_add_c(input, dest, stride, txfm_param);
}

static void inv_txfm_add_32x32(const tran_low_t *input, uint8_t *dest,
                               int stride, const TxfmParam *txfm_param) {
  av1_iht32x32_1024_add_c(input, dest, stride, txfm_param);
}

#if CONFIG_TX64X64
static void inv_txfm_add_64x64(const tran_low_t *input, uint8_t *dest,
                               int stride, const TxfmParam *txfm_param) {
  av1_iht64x64_4096_add_c(input, dest, stride, txfm_param);
}
#endif  // CONFIG_TX64X64

void av1_highbd_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                            int eob, int bd) {
  if (eob > 1)
    aom_highbd_iwht4x4_16_add(input, dest, stride, bd);
  else
    aom_highbd_iwht4x4_1_add(input, dest, stride, bd);
}

#if CONFIG_CHROMA_2X2
static void highbd_inv_txfm_add_2x2(const tran_low_t *input, uint8_t *dest,
                                    int stride, const TxfmParam *txfm_param) {
  int eob = txfm_param->eob;
  int bd = txfm_param->bd;
  int lossless = txfm_param->lossless;
  TX_TYPE tx_type = txfm_param->tx_type;
  tran_high_t a1 = input[0] >> UNIT_QUANT_SHIFT;
  tran_high_t b1 = input[1] >> UNIT_QUANT_SHIFT;
  tran_high_t c1 = input[2] >> UNIT_QUANT_SHIFT;
  tran_high_t d1 = input[3] >> UNIT_QUANT_SHIFT;

  tran_high_t a2 = a1 + c1;
  tran_high_t b2 = b1 + d1;
  tran_high_t c2 = a1 - c1;
  tran_high_t d2 = b1 - d1;

  uint16_t *dst = CONVERT_TO_SHORTPTR(dest);

  (void)tx_type;
  (void)lossless;
  (void)eob;

  a1 = (a2 + b2) >> 2;
  b1 = (a2 - b2) >> 2;
  c1 = (c2 + d2) >> 2;
  d1 = (c2 - d2) >> 2;

  dst[0] = highbd_clip_pixel_add(dst[0], a1, bd);
  dst[1] = highbd_clip_pixel_add(dst[1], b1, bd);
  dst[stride] = highbd_clip_pixel_add(dst[stride], c1, bd);
  dst[stride + 1] = highbd_clip_pixel_add(dst[stride + 1], d1, bd);
}
#endif

void av1_highbd_inv_txfm_add_4x4(const tran_low_t *input, uint8_t *dest,
                                 int stride, const TxfmParam *txfm_param) {
  int eob = txfm_param->eob;
  int bd = txfm_param->bd;
  int lossless = txfm_param->lossless;
  const int32_t *src = (const int32_t *)input;
  TX_TYPE tx_type = txfm_param->tx_type;
  if (lossless) {
    assert(tx_type == DCT_DCT);
    av1_highbd_iwht4x4_add(input, dest, stride, eob, bd);
    return;
  }
  av1_inv_txfm2d_add_4x4_c(src, CONVERT_TO_SHORTPTR(dest), stride, tx_type, bd);
}

void av1_highbd_inv_txfm_add_4x8(const tran_low_t *input, uint8_t *dest,
                                 int stride, const TxfmParam *txfm_param) {
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_4x8_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                           txfm_param->tx_type, txfm_param->bd);
}

void av1_highbd_inv_txfm_add_8x4(const tran_low_t *input, uint8_t *dest,
                                 int stride, const TxfmParam *txfm_param) {
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_8x4_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                           txfm_param->tx_type, txfm_param->bd);
}

static void highbd_inv_txfm_add_8x16(const tran_low_t *input, uint8_t *dest,
                                     int stride, const TxfmParam *txfm_param) {
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_8x16_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                            txfm_param->tx_type, txfm_param->bd);
}

static void highbd_inv_txfm_add_16x8(const tran_low_t *input, uint8_t *dest,
                                     int stride, const TxfmParam *txfm_param) {
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_16x8_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                            txfm_param->tx_type, txfm_param->bd);
}

static void highbd_inv_txfm_add_16x32(const tran_low_t *input, uint8_t *dest,
                                      int stride, const TxfmParam *txfm_param) {
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_16x32_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                             txfm_param->tx_type, txfm_param->bd);
}

static void highbd_inv_txfm_add_32x16(const tran_low_t *input, uint8_t *dest,
                                      int stride, const TxfmParam *txfm_param) {
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_32x16_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                             txfm_param->tx_type, txfm_param->bd);
}

static void highbd_inv_txfm_add_8x8(const tran_low_t *input, uint8_t *dest,
                                    int stride, const TxfmParam *txfm_param) {
  int bd = txfm_param->bd;
  TX_TYPE tx_type = txfm_param->tx_type;
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_8x8_c(src, CONVERT_TO_SHORTPTR(dest), stride, tx_type, bd);
}

static void highbd_inv_txfm_add_16x16(const tran_low_t *input, uint8_t *dest,
                                      int stride, const TxfmParam *txfm_param) {
  int bd = txfm_param->bd;
  TX_TYPE tx_type = txfm_param->tx_type;
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_16x16_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                             tx_type, bd);
}

static void highbd_inv_txfm_add_32x32(const tran_low_t *input, uint8_t *dest,
                                      int stride, const TxfmParam *txfm_param) {
  int bd = txfm_param->bd;
  TX_TYPE tx_type = txfm_param->tx_type;
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_32x32_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                             tx_type, bd);
}

#if CONFIG_TX64X64
static void highbd_inv_txfm_add_64x64(const tran_low_t *input, uint8_t *dest,
                                      int stride, const TxfmParam *txfm_param) {
  int bd = txfm_param->bd;
  TX_TYPE tx_type = txfm_param->tx_type;
  const int32_t *src = (const int32_t *)input;
  av1_inv_txfm2d_add_64x64_c(src, CONVERT_TO_SHORTPTR(dest), stride,
                             DCT_DCT, bd);
}
#endif  // CONFIG_TX64X64

void av1_inv_txfm_add(const tran_low_t *input, uint8_t *dest, int stride,
                      TxfmParam *txfm_param) {
  const TX_SIZE tx_size = txfm_param->tx_size;
  switch (tx_size) {
#if CONFIG_TX64X64
    case TX_64X64: inv_txfm_add_64x64(input, dest, stride, txfm_param); break;
#endif  // CONFIG_TX64X64
    case TX_32X32: inv_txfm_add_32x32(input, dest, stride, txfm_param); break;
    case TX_16X16: inv_txfm_add_16x16(input, dest, stride, txfm_param); break;
    case TX_8X8: inv_txfm_add_8x8(input, dest, stride, txfm_param); break;
    case TX_4X8: inv_txfm_add_4x8(input, dest, stride, txfm_param); break;
    case TX_8X4: inv_txfm_add_8x4(input, dest, stride, txfm_param); break;
    case TX_8X16: inv_txfm_add_8x16(input, dest, stride, txfm_param); break;
    case TX_16X8: inv_txfm_add_16x8(input, dest, stride, txfm_param); break;
    case TX_16X32: inv_txfm_add_16x32(input, dest, stride, txfm_param); break;
    case TX_32X16: inv_txfm_add_32x16(input, dest, stride, txfm_param); break;
    case TX_4X4:
      // this is like av1_short_idct4x4 but has a special case around eob<=1
      // which is significant (not just an optimization) for the lossless
      // case.
      inv_txfm_add_4x4(input, dest, stride, txfm_param);
      break;
#if CONFIG_CHROMA_2X2
    case TX_2X2: inv_txfm_add_2x2(input, dest, stride, txfm_param); break;
#endif
#if CONFIG_RECT_TX_EXT && (CONFIG_EXT_TX || CONFIG_VAR_TX)
    case TX_32X8: inv_txfm_add_32x8(input, dest, stride, txfm_param); break;
    case TX_8X32: inv_txfm_add_8x32(input, dest, stride, txfm_param); break;
    case TX_16X4: inv_txfm_add_16x4(input, dest, stride, txfm_param); break;
    case TX_4X16: inv_txfm_add_4x16(input, dest, stride, txfm_param); break;
#endif
    default: assert(0 && "Invalid transform size"); break;
  }
}

static void init_txfm_param(const MACROBLOCKD *xd, TX_SIZE tx_size,
                            TX_TYPE tx_type, int eob, TxfmParam *txfm_param) {
  txfm_param->tx_type = tx_type;
  txfm_param->tx_size = tx_size;
  txfm_param->eob = eob;
  txfm_param->lossless = xd->lossless[xd->mi[0]->mbmi.segment_id];
  txfm_param->bd = xd->bd;
#if CONFIG_LGT
  txfm_param->is_inter = is_inter_block(&xd->mi[0]->mbmi);
#endif
#if CONFIG_ADAPT_SCAN
  txfm_param->eob_threshold =
      (const int16_t *)&xd->eob_threshold_md[tx_size][tx_type][0];
#endif
}

#if !CONFIG_TXMG
typedef void (*InvTxfmFunc)(const tran_low_t *dqcoeff, uint8_t *dst, int stride,
                            TxfmParam *txfm_param);

static InvTxfmFunc inv_txfm_func[2] = { av1_inv_txfm_add,
                                        av1_highbd_inv_txfm_add };
#endif

void av1_inverse_transform_block(const MACROBLOCKD *xd,
                                 const tran_low_t *dqcoeff,
#if CONFIG_LGT
                                 PREDICTION_MODE mode,
#endif
#if CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
                                 uint8_t *mrc_mask,
#endif  // CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
                                 TX_TYPE tx_type, TX_SIZE tx_size, uint8_t *dst,
                                 int stride, int eob) {
  if (!eob) return;
#if CONFIG_PVQ
  const BLOCK_SIZE tx_bsize = txsize_to_bsize[tx_size];
  const int txb_width = block_size_wide[tx_bsize];
  const int txb_height = block_size_high[tx_bsize];
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    for (int r = 0; r < txb_height; r++)
      for (int c = 0; c < txb_width; c++)
        CONVERT_TO_SHORTPTR(dst)[r * stride + c] = 0;
  } else {
    for (int r = 0; r < txb_height; r++)
      for (int c = 0; c < txb_width; c++) dst[r * stride + c] = 0;
  }
#endif  // CONFIG_PVQ
  TxfmParam txfm_param;
  init_txfm_param(xd, tx_size, tx_type, eob, &txfm_param);
#if CONFIG_LGT || CONFIG_MRC_TX
  txfm_param.is_inter = is_inter_block(&xd->mi[0]->mbmi);
  txfm_param.dst = dst;
  txfm_param.stride = stride;
#if CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
  txfm_param.mask = mrc_mask;
#endif  // CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
#if CONFIG_LGT
  txfm_param.mode = mode;
#endif  // CONFIG_LGT
#endif  // CONFIG_LGT || CONFIG_MRC_TX

  const int is_hbd = get_bitdepth_data_path_index(xd);
#if CONFIG_TXMG
  if (is_hbd) {
    av1_highbd_inv_txfm_add(dqcoeff, dst, stride, &txfm_param);
  } else {
    DECLARE_ALIGNED(16, uint16_t, tmp[MAX_TX_SQUARE]);
    int tmp_stride = MAX_TX_SIZE;
    int w = tx_size_wide[tx_size];
    int h = tx_size_high[tx_size];
    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        tmp[r * tmp_stride + c] = dst[r * stride + c];
      }
    }

    av1_highbd_inv_txfm_add(dqcoeff, CONVERT_TO_BYTEPTR(tmp), tmp_stride,
                            &txfm_param);

    for (int r = 0; r < h; ++r) {
      for (int c = 0; c < w; ++c) {
        dst[r * stride + c] = tmp[r * tmp_stride + c];
      }
    }
  }
#else  // CONFIG_TXMG
  inv_txfm_func[is_hbd](dqcoeff, dst, stride, &txfm_param);
#endif  // CONFIG_TXMG
}

void av1_inverse_transform_block_facade(MACROBLOCKD *xd, int plane, int block,
                                        int blk_row, int blk_col, int eob) {
  struct macroblockd_plane *const pd = &xd->plane[plane];
  tran_low_t *dqcoeff = BLOCK_OFFSET(pd->dqcoeff, block);
#if CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
  uint8_t *mrc_mask = BLOCK_OFFSET(xd->mrc_mask, block);
#endif  // CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
  const PLANE_TYPE plane_type = get_plane_type(plane);
  const TX_SIZE tx_size = av1_get_tx_size(plane, xd);
  const TX_TYPE tx_type =
      av1_get_tx_type(plane_type, xd, blk_row, blk_col, block, tx_size);
  const int dst_stride = pd->dst.stride;
  uint8_t *dst =
      &pd->dst.buf[(blk_row * dst_stride + blk_col) << tx_size_wide_log2[0]];
  av1_inverse_transform_block(xd, dqcoeff,
#if CONFIG_LGT
                              xd->mi[0]->mbmi.mode,
#endif  // CONFIG_LGT
#if CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
                              mrc_mask,
#endif  // CONFIG_MRC_TX && SIGNAL_ANY_MRC_MASK
                              tx_type, tx_size, dst, dst_stride, eob);
}

void av1_highbd_inv_txfm_add(const tran_low_t *input, uint8_t *dest, int stride,
                             TxfmParam *txfm_param) {
  const TX_SIZE tx_size = txfm_param->tx_size;
  switch (tx_size) {
#if CONFIG_TX64X64
    case TX_64X64:
      highbd_inv_txfm_add_64x64(input, dest, stride, txfm_param);
      break;
#endif  // CONFIG_TX64X64
    case TX_32X32:
      highbd_inv_txfm_add_32x32(input, dest, stride, txfm_param);
      break;
    case TX_16X16:
      highbd_inv_txfm_add_16x16(input, dest, stride, txfm_param);
      break;
    case TX_8X8:
      highbd_inv_txfm_add_8x8(input, dest, stride, txfm_param);
      break;
    case TX_4X8:
      av1_highbd_inv_txfm_add_4x8(input, dest, stride, txfm_param);
      break;
    case TX_8X4:
      av1_highbd_inv_txfm_add_8x4(input, dest, stride, txfm_param);
      break;
    case TX_8X16:
      highbd_inv_txfm_add_8x16(input, dest, stride, txfm_param);
      break;
    case TX_16X8:
      highbd_inv_txfm_add_16x8(input, dest, stride, txfm_param);
      break;
    case TX_16X32:
      highbd_inv_txfm_add_16x32(input, dest, stride, txfm_param);
      break;
    case TX_32X16:
      highbd_inv_txfm_add_32x16(input, dest, stride, txfm_param);
      break;
    case TX_4X4:
      // this is like av1_short_idct4x4 but has a special case around eob<=1
      // which is significant (not just an optimization) for the lossless
      // case.
      av1_highbd_inv_txfm_add_4x4(input, dest, stride, txfm_param);
      break;
#if CONFIG_CHROMA_2X2
    case TX_2X2:
      highbd_inv_txfm_add_2x2(input, dest, stride, txfm_param);
      break;
#endif
    default: assert(0 && "Invalid transform size"); break;
  }
}
