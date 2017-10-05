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
#include "aom_dsp/quantize.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"

#include "av1/common/idct.h"
#include "av1/common/quant_common.h"
#include "av1/common/scan.h"
#include "av1/common/seg_common.h"

#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/rd.h"

void av1_quantize_skip(intptr_t n_coeffs, tran_low_t *qcoeff_ptr,
                       tran_low_t *dqcoeff_ptr, uint16_t *eob_ptr) {
  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));
  *eob_ptr = 0;
}

static void quantize_fp_helper_c(
    const tran_low_t *coeff_ptr, intptr_t n_coeffs, int skip_block,
    const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr, int log_scale) {
  int i, eob = -1;
  // TODO(jingning) Decide the need of these arguments after the
  // quantization process is completed.
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  (void)iscan;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  if (!skip_block) {
    // Quantization pass: All coefficients with index >= zero_flag are
    // skippable. Note: zero_flag can be zero.
    for (i = 0; i < n_coeffs; i++) {
      const int rc = scan[i];
      const int coeff = coeff_ptr[rc];
      const qm_val_t wt = qm_ptr ? qm_ptr[rc] : (1 << AOM_QM_BITS);
      const qm_val_t iwt = iqm_ptr ? iqm_ptr[rc] : (1 << AOM_QM_BITS);
      const int dequant =
          (dequant_ptr[rc != 0] * iwt + (1 << (AOM_QM_BITS - 1))) >>
          AOM_QM_BITS;
      const int coeff_sign = (coeff >> 31);
      int64_t abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
      int tmp32 = 0;
      if (abs_coeff * wt >=
          (dequant_ptr[rc != 0] << (AOM_QM_BITS - (1 + log_scale)))) {
        abs_coeff += ROUND_POWER_OF_TWO(round_ptr[rc != 0], log_scale);
        abs_coeff = clamp64(abs_coeff, INT16_MIN, INT16_MAX);
        tmp32 = (int)((abs_coeff * wt * quant_ptr[rc != 0]) >>
                      (16 - log_scale + AOM_QM_BITS));
        qcoeff_ptr[rc] = (tmp32 ^ coeff_sign) - coeff_sign;
        dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant / (1 << log_scale);
      }

      if (tmp32) eob = i;
    }
  }
  *eob_ptr = eob + 1;
}

static void highbd_quantize_fp_helper_c(
    const tran_low_t *coeff_ptr, intptr_t count, int skip_block,
    const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, const qm_val_t *qm_ptr,
    const qm_val_t *iqm_ptr, int log_scale) {
  int i;
  int eob = -1;
  const int scale = 1 << log_scale;
  const int shift = 16 - log_scale;
  // TODO(jingning) Decide the need of these arguments after the
  // quantization process is completed.
  (void)zbin_ptr;
  (void)quant_shift_ptr;
  (void)iscan;

  memset(qcoeff_ptr, 0, count * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, count * sizeof(*dqcoeff_ptr));

  if (!skip_block) {
    // Quantization pass: All coefficients with index >= zero_flag are
    // skippable. Note: zero_flag can be zero.
    for (i = 0; i < count; i++) {
      const int rc = scan[i];
      const int coeff = coeff_ptr[rc];
      const qm_val_t wt = qm_ptr != NULL ? qm_ptr[rc] : (1 << AOM_QM_BITS);
      const qm_val_t iwt = iqm_ptr != NULL ? iqm_ptr[rc] : (1 << AOM_QM_BITS);
      const int dequant =
          (dequant_ptr[rc != 0] * iwt + (1 << (AOM_QM_BITS - 1))) >>
          AOM_QM_BITS;
      const int coeff_sign = (coeff >> 31);
      const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
      const int64_t tmp = abs_coeff + (round_ptr[rc != 0] >> log_scale);
      const int abs_qcoeff =
          (int)((tmp * quant_ptr[rc != 0] * wt) >> (shift + AOM_QM_BITS));
      qcoeff_ptr[rc] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
      dqcoeff_ptr[rc] = qcoeff_ptr[rc] * dequant / scale;
      if (abs_qcoeff) eob = i;
    }
  }
  *eob_ptr = eob + 1;
}

void av1_quantize_fp_c(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                       int skip_block, const int16_t *zbin_ptr,
                       const int16_t *round_ptr, const int16_t *quant_ptr,
                       const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
                       tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,
                       uint16_t *eob_ptr, const int16_t *scan,
                       const int16_t *iscan) {
  quantize_fp_helper_c(coeff_ptr, n_coeffs, skip_block, zbin_ptr, round_ptr,
                       quant_ptr, quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,
                       dequant_ptr, eob_ptr, scan, iscan, NULL, NULL, 0);
}

void av1_quantize_fp_facade(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                            const MACROBLOCK_PLANE *p, tran_low_t *qcoeff_ptr,
                            const MACROBLOCKD_PLANE *pd,
                            tran_low_t *dqcoeff_ptr, uint16_t *eob_ptr,
                            const SCAN_ORDER *sc, const QUANT_PARAM *qparam) {
  // obsolete skip_block
  const int skip_block = 0;
  (void)pd;
#if CONFIG_AOM_QM
  const qm_val_t *qm_ptr = qparam->qmatrix;
  const qm_val_t *iqm_ptr = qparam->iqmatrix;
  if (qm_ptr != NULL && iqm_ptr != NULL) {
    quantize_fp_helper_c(coeff_ptr, n_coeffs, skip_block, p->zbinTX,
                         p->round_fpTX, p->quant_fpTX, p->quant_shiftTX,
                         qcoeff_ptr, dqcoeff_ptr,
                         p->dequantTX, eob_ptr, sc->scan, sc->iscan, qm_ptr,
                         iqm_ptr, qparam->log_scale);
  } else {
#endif
    quantize_fp_helper_c(coeff_ptr, n_coeffs, skip_block,
                         p->zbinTX, p->round_fpTX, p->quant_fpTX,
                         p->quant_shiftTX, qcoeff_ptr, dqcoeff_ptr,
                         p->dequantTX, eob_ptr, sc->scan, sc->iscan,
                         NULL, NULL, qparam->log_scale);
#if CONFIG_AOM_QM
  }
#endif
}

void av1_quantize_b_facade(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                           const MACROBLOCK_PLANE *p, tran_low_t *qcoeff_ptr,
                           const MACROBLOCKD_PLANE *pd, tran_low_t *dqcoeff_ptr,
                           uint16_t *eob_ptr, const SCAN_ORDER *sc,
                           const QUANT_PARAM *qparam) {
  // obsolete skip_block
  const int skip_block = 0;
  (void)pd;
#if CONFIG_AOM_QM
  const qm_val_t *qm_ptr = qparam->qmatrix;
  const qm_val_t *iqm_ptr = qparam->iqmatrix;
  if (qm_ptr != NULL && iqm_ptr != NULL) {
    quantize_b_helper_c(coeff_ptr, n_coeffs, skip_block, p->zbinTX, p->roundTX,
                        p->quantTX, p->quant_shiftTX, qcoeff_ptr, dqcoeff_ptr,
                        p->dequantTX, eob_ptr, sc->scan, sc->iscan, qm_ptr,
                        iqm_ptr, qparam->log_scale);
  } else {
#endif  // CONFIG_AOM_QM
    quantize_b_helper_c(coeff_ptr, n_coeffs, skip_block, p->zbinTX, p->roundTX,
                        p->quantTX, p->quant_shiftTX, qcoeff_ptr, dqcoeff_ptr,
                        p->dequantTX, eob_ptr, sc->scan, sc->iscan, qm_ptr,
                        iqm_ptr, qparam->log_scale);
#if CONFIG_AOM_QM
  }
#endif
}

static void quantize_dc(const tran_low_t *coeff_ptr, int n_coeffs,
                        int skip_block, const int16_t *round_ptr,
                        const int16_t quant, tran_low_t *qcoeff_ptr,
                        tran_low_t *dqcoeff_ptr, const int16_t dequant_ptr,
                        uint16_t *eob_ptr, const qm_val_t *qm_ptr,
                        const qm_val_t *iqm_ptr, const int log_scale) {
  const int rc = 0;
  const int coeff = coeff_ptr[rc];
  const int coeff_sign = (coeff >> 31);
  const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
  int64_t tmp;
  int eob = -1;
  int32_t tmp32;
  int dequant;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  if (!skip_block) {
    const int wt = qm_ptr != NULL ? qm_ptr[rc] : (1 << AOM_QM_BITS);
    const int iwt = iqm_ptr != NULL ? iqm_ptr[rc] : (1 << AOM_QM_BITS);
    tmp = clamp(abs_coeff + ROUND_POWER_OF_TWO(round_ptr[rc != 0], log_scale),
                INT16_MIN, INT16_MAX);
    tmp32 = (int32_t)((tmp * wt * quant) >> (16 - log_scale + AOM_QM_BITS));
    qcoeff_ptr[rc] = (tmp32 ^ coeff_sign) - coeff_sign;
    dequant = (dequant_ptr * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;
    dqcoeff_ptr[rc] = (qcoeff_ptr[rc] * dequant) / (1 << log_scale);
    if (tmp32) eob = 0;
  }
  *eob_ptr = eob + 1;
}

void av1_quantize_dc_facade(const tran_low_t *coeff_ptr, intptr_t n_coeffs,
                            const MACROBLOCK_PLANE *p, tran_low_t *qcoeff_ptr,
                            const MACROBLOCKD_PLANE *pd,
                            tran_low_t *dqcoeff_ptr, uint16_t *eob_ptr,
                            const SCAN_ORDER *sc, const QUANT_PARAM *qparam) {
  // obsolete skip_block
  const int skip_block = 0;
  (void)sc;
  (void)pd;
  assert(qparam->log_scale >= 0 && qparam->log_scale < (2 + CONFIG_TX64X64));
#if CONFIG_AOM_QM
  const qm_val_t *qm_ptr = qparam->qmatrix;
  const qm_val_t *iqm_ptr = qparam->iqmatrix;
#else
  const qm_val_t *qm_ptr = NULL;
  const qm_val_t *iqm_ptr = NULL;
#endif
  quantize_dc(coeff_ptr, (int)n_coeffs, skip_block,
              p->roundTX, p->quant_fpTX[0],
              qcoeff_ptr, dqcoeff_ptr, p->dequantTX[0],
              eob_ptr, qm_ptr, iqm_ptr,
              qparam->log_scale);
}

void av1_highbd_quantize_fp_facade(const tran_low_t *coeff_ptr,
                                   intptr_t n_coeffs, const MACROBLOCK_PLANE *p,
                                   tran_low_t *qcoeff_ptr,
                                   const MACROBLOCKD_PLANE *pd,
                                   tran_low_t *dqcoeff_ptr, uint16_t *eob_ptr,
                                   const SCAN_ORDER *sc,
                                   const QUANT_PARAM *qparam) {
  // obsolete skip_block
  const int skip_block = 0;
  (void)pd;
#if CONFIG_AOM_QM
  const qm_val_t *qm_ptr = qparam->qmatrix;
  const qm_val_t *iqm_ptr = qparam->iqmatrix;
  if (qm_ptr != NULL && iqm_ptr != NULL) {
    highbd_quantize_fp_helper_c(coeff_ptr, n_coeffs, skip_block,
                                p->zbinTX, p->round_fpTX, p->quant_fpTX,
                                p->quant_shiftTX,
                                qcoeff_ptr, dqcoeff_ptr,
                                p->dequantTX, eob_ptr, sc->scan,
                                sc->iscan, qm_ptr, iqm_ptr, qparam->log_scale);
  } else {
#endif  // CONFIG_AOM_QM
    av1_highbd_quantize_fp_c(coeff_ptr, n_coeffs, skip_block, p->zbinTX,
                             p->round_fpTX, p->quant_fpTX, p->quant_shiftTX,
                             qcoeff_ptr, dqcoeff_ptr, p->dequantTX, eob_ptr,
                             sc->scan, sc->iscan, qparam->log_scale);

#if CONFIG_AOM_QM
  }
#endif
}

void av1_highbd_quantize_b_facade(const tran_low_t *coeff_ptr,
                                  intptr_t n_coeffs, const MACROBLOCK_PLANE *p,
                                  tran_low_t *qcoeff_ptr,
                                  const MACROBLOCKD_PLANE *pd,
                                  tran_low_t *dqcoeff_ptr, uint16_t *eob_ptr,
                                  const SCAN_ORDER *sc,
                                  const QUANT_PARAM *qparam) {
  // obsolete skip_block
  const int skip_block = 0;
  (void)pd;
#if CONFIG_AOM_QM
  const qm_val_t *qm_ptr = qparam->qmatrix;
  const qm_val_t *iqm_ptr = qparam->iqmatrix;
  if (qm_ptr != NULL && iqm_ptr != NULL) {
    highbd_quantize_b_helper_c(coeff_ptr, n_coeffs, skip_block, p->zbinTX,
                               p->roundTX, p->quantTX, p->quant_shiftTX,
                               qcoeff_ptr, dqcoeff_ptr, p->dequantTX,
                               eob_ptr, sc->scan,
                               sc->iscan, qm_ptr, iqm_ptr, qparam->log_scale);
  } else {
#endif  // CONFIG_AOM_QM
    aom_highbd_quantize_b_c(coeff_ptr, n_coeffs, skip_block, p->zbinTX,
                            p->roundTX, p->quantTX, p->quant_shiftTX,
                            qcoeff_ptr, dqcoeff_ptr, p->dequantTX, eob_ptr,
                            sc->scan, sc->iscan);
#if CONFIG_AOM_QM
  }
#endif
}

static INLINE void highbd_quantize_dc(
    const tran_low_t *coeff_ptr, int n_coeffs, int skip_block,
    const int16_t *round_ptr, const int16_t quant, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t dequant_ptr, uint16_t *eob_ptr,
    const qm_val_t *qm_ptr, const qm_val_t *iqm_ptr, const int log_scale) {
  int eob = -1;

  memset(qcoeff_ptr, 0, n_coeffs * sizeof(*qcoeff_ptr));
  memset(dqcoeff_ptr, 0, n_coeffs * sizeof(*dqcoeff_ptr));

  if (!skip_block) {
    const qm_val_t wt = qm_ptr != NULL ? qm_ptr[0] : (1 << AOM_QM_BITS);
    const qm_val_t iwt = iqm_ptr != NULL ? iqm_ptr[0] : (1 << AOM_QM_BITS);
    const int coeff = coeff_ptr[0];
    const int coeff_sign = (coeff >> 31);
    const int abs_coeff = (coeff ^ coeff_sign) - coeff_sign;
    const int64_t tmp = abs_coeff + ROUND_POWER_OF_TWO(round_ptr[0], log_scale);
    const int64_t tmpw = tmp * wt;
    const int abs_qcoeff =
        (int)((tmpw * quant) >> (16 - log_scale + AOM_QM_BITS));
    qcoeff_ptr[0] = (tran_low_t)((abs_qcoeff ^ coeff_sign) - coeff_sign);
    const int dequant =
        (dequant_ptr * iwt + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS;

    dqcoeff_ptr[0] = (qcoeff_ptr[0] * dequant) / (1 << log_scale);
    if (abs_qcoeff) eob = 0;
  }
  *eob_ptr = eob + 1;
}

void av1_highbd_quantize_dc_facade(const tran_low_t *coeff_ptr,
                                   intptr_t n_coeffs, const MACROBLOCK_PLANE *p,
                                   tran_low_t *qcoeff_ptr,
                                   const MACROBLOCKD_PLANE *pd,
                                   tran_low_t *dqcoeff_ptr, uint16_t *eob_ptr,
                                   const SCAN_ORDER *sc,
                                   const QUANT_PARAM *qparam) {
  // obsolete skip_block
  const int skip_block = 0;
  (void)pd;
#if CONFIG_AOM_QM
  const qm_val_t *qm_ptr = qparam->qmatrix;
  const qm_val_t *iqm_ptr = qparam->iqmatrix;
#else
  const qm_val_t *qm_ptr = NULL;
  const qm_val_t *iqm_ptr = NULL;
#endif  // CONFIG_AOM_QM

  (void)sc;

  highbd_quantize_dc(coeff_ptr, (int)n_coeffs, skip_block, p->roundTX,
                     p->quant_fpTX[0], qcoeff_ptr, dqcoeff_ptr, p->dequantTX[0],
                     eob_ptr, qm_ptr, iqm_ptr, qparam->log_scale);
}

void av1_highbd_quantize_fp_c(
    const tran_low_t *coeff_ptr, intptr_t count, int skip_block,
    const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr,
    const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr,
    tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr,
    const int16_t *scan, const int16_t *iscan, int log_scale) {
  highbd_quantize_fp_helper_c(coeff_ptr, count, skip_block, zbin_ptr, round_ptr,
                              quant_ptr, quant_shift_ptr, qcoeff_ptr,
                              dqcoeff_ptr, dequant_ptr, eob_ptr, scan, iscan,
                              NULL, NULL, log_scale);
}

static void invert_quant(int16_t *quant, int16_t *shift, int d) {
  uint32_t t;
  int l, m;
  t = d;
  for (l = 0; t > 1; l++) t >>= 1;
  m = 1 + (1 << (16 + l)) / d;
  *quant = (int16_t)(m - (1 << 16));
  *shift = 1 << (16 - l);
}

static int get_qzbin_factor(int q, aom_bit_depth_t bit_depth) {
  const int quant = av1_dc_quant(q, 0, bit_depth);
#if CONFIG_HIGHBITDEPTH
  switch (bit_depth) {
    case AOM_BITS_8: return q == 0 ? 64 : (quant < 148 ? 84 : 80);
    case AOM_BITS_10: return q == 0 ? 64 : (quant < 592 ? 84 : 80);
    case AOM_BITS_12: return q == 0 ? 64 : (quant < 2368 ? 84 : 80);
    default:
      assert(0 && "bit_depth should be AOM_BITS_8, AOM_BITS_10 or AOM_BITS_12");
      return -1;
  }
#else
  (void)bit_depth;
  return q == 0 ? 64 : (quant < 148 ? 84 : 80);
#endif
}

void av1_build_quantizer(aom_bit_depth_t bit_depth, int y_dc_delta_q,
                         int uv_dc_delta_q, int uv_ac_delta_q,
                         QUANTS *const quants, Dequants *const deq) {
  int i, q, quant3, quantTX, depth_scale;
  depth_scale = TX_COEFF_DEPTH - bit_depth - 3;

  for (q = 0; q < QINDEX_RANGE; q++) {
    const int qzbin_factor = get_qzbin_factor(q, bit_depth);
    const int qrounding_factor = q == 0 ? 64 : 48;

    for (i = 0; i < 2; ++i) {
      int qrounding_factor_fp = 64;
      // y quantizer setup with original coeef shift of 3
      quant3 = i == 0 ? av1_dc_quant(q, y_dc_delta_q, bit_depth)
        : av1_ac_quant(q, 0, bit_depth);
      // quantizer with TX scaling
      if (depth_scale > 0) {
        quantTX = quant3 << depth_scale;
      }else{
        // this is bull, but it prevents major breakage
        int depth_round = depth_scale < -1 ? 1 << (-depth_scale-1) : 0;
        quantTX = (quant3 + depth_round) >> depth_scale;
        if (quantTX < 4) quantTX = 4;
      }

      invert_quant(&quants->y_quant[q][i],
                   &quants->y_quant_shift[q][i],
                   quantTX);
      quants->y_quant_fp[q][i] = (1 << 16) / quantTX;
      quants->y_round_fp[q][i] = (qrounding_factor_fp * quantTX) >> 7;
      quants->y_zbin[q][i] = ROUND_POWER_OF_TWO(qzbin_factor * quantTX, 7);
      quants->y_round[q][i] = (qrounding_factor * quantTX) >> 7;
      deq->y_dequantTX[q][i] = quantTX;
      deq->y_dequant3[q][i] = quant3; // original Q3(D11) scaling, not quantTX

      // uv quantizer setup with original coeef shift of 3
      quant3 = i == 0 ? av1_dc_quant(q, uv_dc_delta_q, bit_depth)
                     : av1_ac_quant(q, uv_ac_delta_q, bit_depth);
      if (depth_scale > 0) {
        quantTX = quant3 << depth_scale;
      }else{
        // this is bull, but it prevents major breakage
        int depth_round = depth_scale < -1 ? 1 << (-depth_scale-1) : 0;
        quantTX = (quant3 + depth_round) >> depth_scale;
        if (quantTX < 4) quantTX = 4;
      }
      invert_quant(&quants->uv_quant[q][i], &quants->uv_quant_shift[q][i],
                   quantTX);
      quants->uv_quant_fp[q][i] = (1 << 16) / quantTX;
      quants->uv_round_fp[q][i] = (qrounding_factor_fp * quantTX) >> 7;
      quants->uv_zbin[q][i] = ROUND_POWER_OF_TWO(qzbin_factor * quantTX, 7);
      quants->uv_round[q][i] = (qrounding_factor * quantTX) >> 7;
      deq->uv_dequantTX[q][i] = quantTX;
      deq->uv_dequant3[q][i] = quant3;
    }

    for (i = 2; i < 8; i++) {  // 8: SIMD width
      quants->y_quant[q][i] = quants->y_quant[q][1];
      quants->y_quant_fp[q][i] = quants->y_quant_fp[q][1];
      quants->y_round_fp[q][i] = quants->y_round_fp[q][1];
      quants->y_quant_shift[q][i] = quants->y_quant_shift[q][1];
      quants->y_zbin[q][i] = quants->y_zbin[q][1];
      quants->y_round[q][i] = quants->y_round[q][1];
      deq->y_dequantTX[q][i] = deq->y_dequantTX[q][1];
      deq->y_dequant3[q][i] = deq->y_dequant3[q][1];

      quants->uv_quant[q][i] = quants->uv_quant[q][1];
      quants->uv_quant_fp[q][i] = quants->uv_quant_fp[q][1];
      quants->uv_round_fp[q][i] = quants->uv_round_fp[q][1];
      quants->uv_quant_shift[q][i] = quants->uv_quant_shift[q][1];
      quants->uv_zbin[q][i] = quants->uv_zbin[q][1];
      quants->uv_round[q][i] = quants->uv_round[q][1];
      deq->uv_dequantTX[q][i] = deq->uv_dequantTX[q][1];
      deq->uv_dequant3[q][i] = deq->uv_dequant3[q][1];
    }
  }
}

void av1_init_quantizer(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  QUANTS *const quants = &cpi->quants;
  Dequants *const dequants = &cpi->dequants;
  av1_build_quantizer(cm->bit_depth, cm->y_dc_delta_q, cm->uv_dc_delta_q,
                      cm->uv_ac_delta_q, quants, dequants);
}

void av1_init_plane_quantizers(const AV1_COMP *cpi, MACROBLOCK *x,
                               int segment_id) {
  const AV1_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const QUANTS *const quants = &cpi->quants;

#if CONFIG_DELTA_Q
#if CONFIG_EXT_DELTA_Q
  int current_q_index =
      AOMMAX(0, AOMMIN(QINDEX_RANGE - 1,
                       cpi->oxcf.deltaq_mode != NO_DELTA_Q
                           ? cm->base_qindex + xd->delta_qindex
                           : cm->base_qindex));
#else
  int current_q_index = AOMMAX(
      0, AOMMIN(QINDEX_RANGE - 1,
                cm->delta_q_present_flag ? cm->base_qindex + xd->delta_qindex
                                         : cm->base_qindex));
#endif
  const int qindex = av1_get_qindex(&cm->seg, segment_id, current_q_index);
#else
  const int qindex = av1_get_qindex(&cm->seg, segment_id, cm->base_qindex);
#endif
  const int rdmult = av1_compute_rd_mult(cpi, qindex + cm->y_dc_delta_q);
  int i;
#if CONFIG_AOM_QM
  int minqm = cm->min_qmlevel;
  int maxqm = cm->max_qmlevel;
  // Quant matrix only depends on the base QP so there is only one set per frame
  int qmlevel = (xd->lossless[segment_id] || cm->using_qmatrix == 0)
                    ? NUM_QM_LEVELS - 1
                    : aom_get_qmlevel(cm->base_qindex, minqm, maxqm);
#endif

  // Y
  x->plane[0].quantTX = quants->y_quant[qindex];
  x->plane[0].quant_fpTX = quants->y_quant_fp[qindex];
  x->plane[0].round_fpTX = quants->y_round_fp[qindex];
  x->plane[0].quant_shiftTX = quants->y_quant_shift[qindex];
  x->plane[0].zbinTX = quants->y_zbin[qindex];
  x->plane[0].roundTX = quants->y_round[qindex];
  x->plane[0].dequantTX = cpi->dequants.y_dequantTX[qindex];
#if CONFIG_AOM_QM
  memcpy(&xd->plane[0].seg_qmatrix[segment_id], cm->gqmatrix[qmlevel][0],
         sizeof(cm->gqmatrix[qmlevel][0]));
  memcpy(&xd->plane[0].seg_iqmatrix[segment_id], cm->giqmatrix[qmlevel][0],
         sizeof(cm->giqmatrix[qmlevel][0]));
#endif
  xd->plane[0].dequant3 = cpi->dequants.y_dequant3[qindex];

  // UV
  for (i = 1; i < 3; i++) {
    x->plane[i].quantTX = quants->uv_quant[qindex];
    x->plane[i].quant_fpTX = quants->uv_quant_fp[qindex];
    x->plane[i].round_fpTX = quants->uv_round_fp[qindex];
    x->plane[i].quant_shiftTX = quants->uv_quant_shift[qindex];
    x->plane[i].zbinTX = quants->uv_zbin[qindex];
    x->plane[i].roundTX = quants->uv_round[qindex];
    x->plane[i].dequantTX = cpi->dequants.uv_dequantTX[qindex];
#if CONFIG_AOM_QM
    memcpy(&xd->plane[i].seg_qmatrix[segment_id], cm->gqmatrix[qmlevel][1],
           sizeof(cm->gqmatrix[qmlevel][1]));
    memcpy(&xd->plane[i].seg_iqmatrix[segment_id], cm->giqmatrix[qmlevel][1],
           sizeof(cm->giqmatrix[qmlevel][1]));
#endif
    xd->plane[i].dequant3 = cpi->dequants.uv_dequant3[qindex];
  }

  x->skip_block = segfeature_active(&cm->seg, segment_id, SEG_LVL_SKIP);
  x->qindex = qindex;

  set_error_per_bit(x, rdmult);

  av1_initialize_me_consts(cpi, x, qindex);
}

void av1_frame_init_quantizer(AV1_COMP *cpi) {
  MACROBLOCK *const x = &cpi->td.mb;
  MACROBLOCKD *const xd = &x->e_mbd;
  av1_init_plane_quantizers(cpi, x, xd->mi[0]->mbmi.segment_id);
}

void av1_set_quantizer(AV1_COMMON *cm, int q) {
  // quantizer has to be reinitialized with av1_init_quantizer() if any
  // delta_q changes.
  cm->base_qindex = q;
  cm->y_dc_delta_q = 0;
  cm->uv_dc_delta_q = 0;
  cm->uv_ac_delta_q = 0;
}

// Table that converts 0-63 Q-range values passed in outside to the Qindex
// range used internally.
static const int quantizer_to_qindex[] = {
  0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,
  52,  56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96,  100,
  104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152,
  156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204,
  208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 249, 255,
};

int av1_quantizer_to_qindex(int quantizer) {
  return quantizer_to_qindex[quantizer];
}

int av1_qindex_to_quantizer(int qindex) {
  int quantizer;

  for (quantizer = 0; quantizer < 64; ++quantizer)
    if (quantizer_to_qindex[quantizer] >= qindex) return quantizer;

  return 63;
}
