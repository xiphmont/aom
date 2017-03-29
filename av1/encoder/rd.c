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

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "./av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_mem/aom_mem.h"
#include "aom_ports/bitops.h"
#include "aom_ports/mem.h"
#include "aom_ports/system_state.h"

#include "av1/common/common.h"
#include "av1/common/entropy.h"
#include "av1/common/entropymode.h"
#include "av1/common/mvref_common.h"
#include "av1/common/pred_common.h"
#include "av1/common/quant_common.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/seg_common.h"

#include "av1/encoder/av1_quantize.h"
#include "av1/encoder/cost.h"
#include "av1/encoder/encodemb.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/mcomp.h"
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/tokenize.h"

#define RD_THRESH_POW 1.25

// Factor to weigh the rate for switchable interp filters.
#define SWITCHABLE_INTERP_RATE_FACTOR 1

void av1_rd_cost_reset(RD_COST *rd_cost) {
  rd_cost->rate = INT_MAX;
  rd_cost->dist = INT64_MAX;
  rd_cost->rdcost = INT64_MAX;
}

void av1_rd_cost_init(RD_COST *rd_cost) {
  rd_cost->rate = 0;
  rd_cost->dist = 0;
  rd_cost->rdcost = 0;
}

// The baseline rd thresholds for breaking out of the rd loop for
// certain modes are assumed to be based on 8x8 blocks.
// This table is used to correct for block size.
// The factors here are << 2 (2 = x0.5, 32 = x8 etc).
static const uint8_t rd_thresh_block_size_factor[BLOCK_SIZES] = {
#if CONFIG_CB4X4
  2,  2,  2,
#endif
  2,  3,  3, 4, 6, 6, 8, 12, 12, 16, 24, 24, 32,
#if CONFIG_EXT_PARTITION
  48, 48, 64
#endif  // CONFIG_EXT_PARTITION
};

static void fill_mode_costs(AV1_COMP *cpi) {
  const FRAME_CONTEXT *const fc = cpi->common.fc;
  int i, j;

  for (i = 0; i < INTRA_MODES; ++i)
    for (j = 0; j < INTRA_MODES; ++j)
      av1_cost_tokens(cpi->y_mode_costs[i][j], av1_kf_y_mode_prob[i][j],
                      av1_intra_mode_tree);

  for (i = 0; i < BLOCK_SIZE_GROUPS; ++i)
    av1_cost_tokens(cpi->mbmode_cost[i], fc->y_mode_prob[i],
                    av1_intra_mode_tree);

  for (i = 0; i < INTRA_MODES; ++i)
    av1_cost_tokens(cpi->intra_uv_mode_cost[i], fc->uv_mode_prob[i],
                    av1_intra_mode_tree);

  for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; ++i)
    av1_cost_tokens(cpi->switchable_interp_costs[i],
                    fc->switchable_interp_prob[i], av1_switchable_interp_tree);

#if CONFIG_PALETTE
  for (i = 0; i < PALETTE_BLOCK_SIZES; ++i) {
    av1_cost_tokens(cpi->palette_y_size_cost[i],
                    av1_default_palette_y_size_prob[i], av1_palette_size_tree);
    av1_cost_tokens(cpi->palette_uv_size_cost[i],
                    av1_default_palette_uv_size_prob[i], av1_palette_size_tree);
  }

  for (i = 0; i < PALETTE_SIZES; ++i) {
    for (j = 0; j < PALETTE_COLOR_INDEX_CONTEXTS; ++j) {
      av1_cost_tokens(cpi->palette_y_color_cost[i][j],
                      av1_default_palette_y_color_index_prob[i][j],
                      av1_palette_color_index_tree[i]);
      av1_cost_tokens(cpi->palette_uv_color_cost[i][j],
                      av1_default_palette_uv_color_index_prob[i][j],
                      av1_palette_color_index_tree[i]);
    }
  }
#endif  // CONFIG_PALETTE

  for (i = 0; i < MAX_TX_DEPTH; ++i)
    for (j = 0; j < TX_SIZE_CONTEXTS; ++j)
      av1_cost_tokens(cpi->tx_size_cost[i][j], fc->tx_size_probs[i][j],
                      av1_tx_size_tree[i]);

#if CONFIG_EXT_TX
  for (i = TX_4X4; i < EXT_TX_SIZES; ++i) {
    int s;
    for (s = 1; s < EXT_TX_SETS_INTER; ++s) {
      if (use_inter_ext_tx_for_txsize[s][i]) {
        av1_cost_tokens(cpi->inter_tx_type_costs[s][i],
                        fc->inter_ext_tx_prob[s][i], av1_ext_tx_inter_tree[s]);
      }
    }
    for (s = 1; s < EXT_TX_SETS_INTRA; ++s) {
      if (use_intra_ext_tx_for_txsize[s][i]) {
        for (j = 0; j < INTRA_MODES; ++j)
          av1_cost_tokens(cpi->intra_tx_type_costs[s][i][j],
                          fc->intra_ext_tx_prob[s][i][j],
                          av1_ext_tx_intra_tree[s]);
      }
    }
  }
#else
  for (i = TX_4X4; i < EXT_TX_SIZES; ++i) {
    for (j = 0; j < TX_TYPES; ++j)
      av1_cost_tokens(cpi->intra_tx_type_costs[i][j],
                      fc->intra_ext_tx_prob[i][j], av1_ext_tx_tree);
  }
  for (i = TX_4X4; i < EXT_TX_SIZES; ++i) {
    av1_cost_tokens(cpi->inter_tx_type_costs[i], fc->inter_ext_tx_prob[i],
                    av1_ext_tx_tree);
  }
#endif  // CONFIG_EXT_TX
#if CONFIG_EXT_INTRA
#if CONFIG_INTRA_INTERP
  for (i = 0; i < INTRA_FILTERS + 1; ++i)
    av1_cost_tokens(cpi->intra_filter_cost[i], fc->intra_filter_probs[i],
                    av1_intra_filter_tree);
#endif  // CONFIG_INTRA_INTERP
#endif  // CONFIG_EXT_INTRA
#if CONFIG_LOOP_RESTORATION
  av1_cost_tokens(cpi->switchable_restore_cost, fc->switchable_restore_prob,
                  av1_switchable_restore_tree);
#endif  // CONFIG_LOOP_RESTORATION
#if CONFIG_GLOBAL_MOTION
  av1_cost_tokens(cpi->gmtype_cost, fc->global_motion_types_prob,
                  av1_global_motion_types_tree);
#endif  // CONFIG_GLOBAL_MOTION
}

void av1_fill_token_costs(av1_coeff_cost *c,
                          av1_coeff_probs_model (*p)[PLANE_TYPES]) {
  int i, j, k, l;
  TX_SIZE t;
  for (t = 0; t < TX_SIZES; ++t)
    for (i = 0; i < PLANE_TYPES; ++i)
      for (j = 0; j < REF_TYPES; ++j)
        for (k = 0; k < COEF_BANDS; ++k)
          for (l = 0; l < BAND_COEFF_CONTEXTS(k); ++l) {
            aom_prob probs[ENTROPY_NODES];
            av1_model_to_full_probs(p[t][i][j][k][l], probs);
            av1_cost_tokens((int *)c[t][i][j][k][0][l], probs, av1_coef_tree);
            av1_cost_tokens_skip((int *)c[t][i][j][k][1][l], probs,
                                 av1_coef_tree);
            assert(c[t][i][j][k][0][l][EOB_TOKEN] ==
                   c[t][i][j][k][1][l][EOB_TOKEN]);
          }
}

// Values are now correlated to quantizer.
static int sad_per_bit16lut_8[QINDEX_RANGE];
static int sad_per_bit4lut_8[QINDEX_RANGE];

#if CONFIG_AOM_HIGHBITDEPTH
static int sad_per_bit16lut_10[QINDEX_RANGE];
static int sad_per_bit4lut_10[QINDEX_RANGE];
static int sad_per_bit16lut_12[QINDEX_RANGE];
static int sad_per_bit4lut_12[QINDEX_RANGE];
#endif

static void init_me_luts_bd(int *bit16lut, int *bit4lut, int range,
                            aom_bit_depth_t bit_depth) {
  int i;
  // Initialize the sad lut tables using a formulaic calculation for now.
  // This is to make it easier to resolve the impact of experimental changes
  // to the quantizer tables.
  for (i = 0; i < range; i++) {
    const double q = av1_convert_qindex_to_q(i, bit_depth);
    bit16lut[i] = (int)(0.0418 * q + 2.4107);
    bit4lut[i] = (int)(0.063 * q + 2.742);
  }
}

void av1_init_me_luts(void) {
  init_me_luts_bd(sad_per_bit16lut_8, sad_per_bit4lut_8, QINDEX_RANGE,
                  AOM_BITS_8);
#if CONFIG_AOM_HIGHBITDEPTH
  init_me_luts_bd(sad_per_bit16lut_10, sad_per_bit4lut_10, QINDEX_RANGE,
                  AOM_BITS_10);
  init_me_luts_bd(sad_per_bit16lut_12, sad_per_bit4lut_12, QINDEX_RANGE,
                  AOM_BITS_12);
#endif
}

static const int rd_boost_factor[16] = { 64, 32, 32, 32, 24, 16, 12, 12,
                                         8,  8,  4,  4,  2,  2,  1,  0 };
static const int rd_frame_type_factor[FRAME_UPDATE_TYPES] = {
  128, 144, 128, 128, 144,
#if CONFIG_EXT_REFS
  // TODO(zoeliu): To adjust further following factor values.
  128, 128, 128
  // TODO(weitinglin): We should investigate if the values should be the same
  //                   as the value used by OVERLAY frame
  ,
  144
#endif  // CONFIG_EXT_REFS
};

int av1_compute_rd_mult(const AV1_COMP *cpi, int qindex) {
  const int64_t q = av1_dc_quant(qindex, 0, cpi->common.bit_depth);
#if CONFIG_AOM_HIGHBITDEPTH
  int64_t rdmult = 0;
  switch (cpi->common.bit_depth) {
    case AOM_BITS_8: rdmult = 88 * q * q / 24; break;
    case AOM_BITS_10: rdmult = ROUND_POWER_OF_TWO(88 * q * q / 24, 4); break;
    case AOM_BITS_12: rdmult = ROUND_POWER_OF_TWO(88 * q * q / 24, 8); break;
    default:
      assert(0 && "bit_depth should be AOM_BITS_8, AOM_BITS_10 or AOM_BITS_12");
      return -1;
  }
#else
  int64_t rdmult = 88 * q * q / 24;
#endif  // CONFIG_AOM_HIGHBITDEPTH
  if (cpi->oxcf.pass == 2 && (cpi->common.frame_type != KEY_FRAME)) {
    const GF_GROUP *const gf_group = &cpi->twopass.gf_group;
    const FRAME_UPDATE_TYPE frame_type = gf_group->update_type[gf_group->index];
    const int boost_index = AOMMIN(15, (cpi->rc.gfu_boost / 100));

    rdmult = (rdmult * rd_frame_type_factor[frame_type]) >> 7;
    rdmult += ((rdmult * rd_boost_factor[boost_index]) >> 7);
  }
  if (rdmult < 1) rdmult = 1;
  return (int)rdmult;
}

static int compute_rd_thresh_factor(int qindex, aom_bit_depth_t bit_depth) {
  double q;
#if CONFIG_AOM_HIGHBITDEPTH
  switch (bit_depth) {
    case AOM_BITS_8: q = av1_dc_quant(qindex, 0, AOM_BITS_8) / 4.0; break;
    case AOM_BITS_10: q = av1_dc_quant(qindex, 0, AOM_BITS_10) / 16.0; break;
    case AOM_BITS_12: q = av1_dc_quant(qindex, 0, AOM_BITS_12) / 64.0; break;
    default:
      assert(0 && "bit_depth should be AOM_BITS_8, AOM_BITS_10 or AOM_BITS_12");
      return -1;
  }
#else
  (void)bit_depth;
  q = av1_dc_quant(qindex, 0, AOM_BITS_8) / 4.0;
#endif  // CONFIG_AOM_HIGHBITDEPTH
  // TODO(debargha): Adjust the function below.
  return AOMMAX((int)(pow(q, RD_THRESH_POW) * 5.12), 8);
}

void av1_initialize_me_consts(const AV1_COMP *cpi, MACROBLOCK *x, int qindex) {
#if CONFIG_AOM_HIGHBITDEPTH
  switch (cpi->common.bit_depth) {
    case AOM_BITS_8:
      x->sadperbit16 = sad_per_bit16lut_8[qindex];
      x->sadperbit4 = sad_per_bit4lut_8[qindex];
      break;
    case AOM_BITS_10:
      x->sadperbit16 = sad_per_bit16lut_10[qindex];
      x->sadperbit4 = sad_per_bit4lut_10[qindex];
      break;
    case AOM_BITS_12:
      x->sadperbit16 = sad_per_bit16lut_12[qindex];
      x->sadperbit4 = sad_per_bit4lut_12[qindex];
      break;
    default:
      assert(0 && "bit_depth should be AOM_BITS_8, AOM_BITS_10 or AOM_BITS_12");
  }
#else
  (void)cpi;
  x->sadperbit16 = sad_per_bit16lut_8[qindex];
  x->sadperbit4 = sad_per_bit4lut_8[qindex];
#endif  // CONFIG_AOM_HIGHBITDEPTH
}

static void set_block_thresholds(const AV1_COMMON *cm, RD_OPT *rd) {
  int i, bsize, segment_id;

  for (segment_id = 0; segment_id < MAX_SEGMENTS; ++segment_id) {
    const int qindex =
        clamp(av1_get_qindex(&cm->seg, segment_id, cm->base_qindex) +
                  cm->y_dc_delta_q,
              0, MAXQ);
    const int q = compute_rd_thresh_factor(qindex, cm->bit_depth);

    for (bsize = 0; bsize < BLOCK_SIZES; ++bsize) {
      // Threshold here seems unnecessarily harsh but fine given actual
      // range of values used for cpi->sf.thresh_mult[].
      const int t = q * rd_thresh_block_size_factor[bsize];
      const int thresh_max = INT_MAX / t;

#if CONFIG_CB4X4
      for (i = 0; i < MAX_MODES; ++i)
        rd->threshes[segment_id][bsize][i] = rd->thresh_mult[i] < thresh_max
                                                 ? rd->thresh_mult[i] * t / 4
                                                 : INT_MAX;
#else
      if (bsize >= BLOCK_8X8) {
        for (i = 0; i < MAX_MODES; ++i)
          rd->threshes[segment_id][bsize][i] = rd->thresh_mult[i] < thresh_max
                                                   ? rd->thresh_mult[i] * t / 4
                                                   : INT_MAX;
      } else {
        for (i = 0; i < MAX_REFS; ++i)
          rd->threshes[segment_id][bsize][i] =
              rd->thresh_mult_sub8x8[i] < thresh_max
                  ? rd->thresh_mult_sub8x8[i] * t / 4
                  : INT_MAX;
      }
#endif
    }
  }
}

#if CONFIG_REF_MV
void av1_set_mvcost(MACROBLOCK *x, MV_REFERENCE_FRAME ref_frame, int ref,
                    int ref_mv_idx) {
  MB_MODE_INFO_EXT *mbmi_ext = x->mbmi_ext;
  int8_t rf_type = av1_ref_frame_type(x->e_mbd.mi[0]->mbmi.ref_frame);
  int nmv_ctx = av1_nmv_ctx(mbmi_ext->ref_mv_count[rf_type],
                            mbmi_ext->ref_mv_stack[rf_type], ref, ref_mv_idx);
  (void)ref_frame;
  x->mvcost = x->mv_cost_stack[nmv_ctx];
  x->nmvjointcost = x->nmv_vec_cost[nmv_ctx];
  x->mvsadcost = x->mvcost;
  x->nmvjointsadcost = x->nmvjointcost;
}
#endif

void av1_initialize_rd_consts(AV1_COMP *cpi) {
  AV1_COMMON *const cm = &cpi->common;
  MACROBLOCK *const x = &cpi->td.mb;
  RD_OPT *const rd = &cpi->rd;
  int i;
#if CONFIG_REF_MV
  int nmv_ctx;
#endif

  aom_clear_system_state();

  rd->RDDIV = RDDIV_BITS;  // In bits (to multiply D by 128).
  rd->RDMULT = av1_compute_rd_mult(cpi, cm->base_qindex + cm->y_dc_delta_q);

  set_error_per_bit(x, rd->RDMULT);

  set_block_thresholds(cm, rd);

#if CONFIG_REF_MV
  for (nmv_ctx = 0; nmv_ctx < NMV_CONTEXTS; ++nmv_ctx) {
    av1_build_nmv_cost_table(
        x->nmv_vec_cost[nmv_ctx],
        cm->allow_high_precision_mv ? x->nmvcost_hp[nmv_ctx]
                                    : x->nmvcost[nmv_ctx],
        &cm->fc->nmvc[nmv_ctx], cm->allow_high_precision_mv);
  }
  x->mvcost = x->mv_cost_stack[0];
  x->nmvjointcost = x->nmv_vec_cost[0];
  x->mvsadcost = x->mvcost;
  x->nmvjointsadcost = x->nmvjointcost;
#else
  av1_build_nmv_cost_table(
      x->nmvjointcost, cm->allow_high_precision_mv ? x->nmvcost_hp : x->nmvcost,
      &cm->fc->nmvc, cm->allow_high_precision_mv);
#endif

  if (cpi->oxcf.pass != 1) {
    av1_fill_token_costs(x->token_costs, cm->fc->coef_probs);

    if (cpi->sf.partition_search_type != VAR_BASED_PARTITION ||
        cm->frame_type == KEY_FRAME) {
#if CONFIG_UNPOISON_PARTITION_CTX
      cpi->partition_cost[0][PARTITION_NONE] = INT_MAX;
      cpi->partition_cost[0][PARTITION_HORZ] = INT_MAX;
      cpi->partition_cost[0][PARTITION_VERT] = INT_MAX;
      cpi->partition_cost[0][PARTITION_SPLIT] = 0;
#endif
#if CONFIG_EXT_PARTITION_TYPES
      av1_cost_tokens(cpi->partition_cost[CONFIG_UNPOISON_PARTITION_CTX],
                      cm->fc->partition_prob[0], av1_partition_tree);
      for (i = 1; i < PARTITION_CONTEXTS_PRIMARY; ++i)
        av1_cost_tokens(cpi->partition_cost[CONFIG_UNPOISON_PARTITION_CTX + i],
                        cm->fc->partition_prob[i], av1_ext_partition_tree);
#else
      for (i = 0; i < PARTITION_CONTEXTS_PRIMARY; ++i)
        av1_cost_tokens(cpi->partition_cost[CONFIG_UNPOISON_PARTITION_CTX + i],
                        cm->fc->partition_prob[i], av1_partition_tree);
#endif  // CONFIG_EXT_PARTITION_TYPES
#if CONFIG_UNPOISON_PARTITION_CTX
      for (; i < PARTITION_CONTEXTS_PRIMARY + PARTITION_BLOCK_SIZES; ++i) {
        aom_prob p = cm->fc->partition_prob[i][PARTITION_VERT];
        assert(p > 0);
        cpi->partition_cost[1 + i][PARTITION_NONE] = INT_MAX;
        cpi->partition_cost[1 + i][PARTITION_HORZ] = INT_MAX;
        cpi->partition_cost[1 + i][PARTITION_VERT] = av1_cost_bit(p, 0);
        cpi->partition_cost[1 + i][PARTITION_SPLIT] = av1_cost_bit(p, 1);
      }
      for (; i < PARTITION_CONTEXTS_PRIMARY + 2 * PARTITION_BLOCK_SIZES; ++i) {
        aom_prob p = cm->fc->partition_prob[i][PARTITION_HORZ];
        assert(p > 0);
        cpi->partition_cost[1 + i][PARTITION_NONE] = INT_MAX;
        cpi->partition_cost[1 + i][PARTITION_HORZ] = av1_cost_bit(p, 0);
        cpi->partition_cost[1 + i][PARTITION_VERT] = INT_MAX;
        cpi->partition_cost[1 + i][PARTITION_SPLIT] = av1_cost_bit(p, 1);
      }
#endif
    }

    fill_mode_costs(cpi);

    if (!frame_is_intra_only(cm)) {
#if CONFIG_REF_MV
      for (i = 0; i < NEWMV_MODE_CONTEXTS; ++i) {
        cpi->newmv_mode_cost[i][0] = av1_cost_bit(cm->fc->newmv_prob[i], 0);
        cpi->newmv_mode_cost[i][1] = av1_cost_bit(cm->fc->newmv_prob[i], 1);
      }

      for (i = 0; i < ZEROMV_MODE_CONTEXTS; ++i) {
        cpi->zeromv_mode_cost[i][0] = av1_cost_bit(cm->fc->zeromv_prob[i], 0);
        cpi->zeromv_mode_cost[i][1] = av1_cost_bit(cm->fc->zeromv_prob[i], 1);
      }

      for (i = 0; i < REFMV_MODE_CONTEXTS; ++i) {
        cpi->refmv_mode_cost[i][0] = av1_cost_bit(cm->fc->refmv_prob[i], 0);
        cpi->refmv_mode_cost[i][1] = av1_cost_bit(cm->fc->refmv_prob[i], 1);
      }

      for (i = 0; i < DRL_MODE_CONTEXTS; ++i) {
        cpi->drl_mode_cost0[i][0] = av1_cost_bit(cm->fc->drl_prob[i], 0);
        cpi->drl_mode_cost0[i][1] = av1_cost_bit(cm->fc->drl_prob[i], 1);
      }
#if CONFIG_EXT_INTER
      cpi->new2mv_mode_cost[0] = av1_cost_bit(cm->fc->new2mv_prob, 0);
      cpi->new2mv_mode_cost[1] = av1_cost_bit(cm->fc->new2mv_prob, 1);
#endif  // CONFIG_EXT_INTER
#else
      for (i = 0; i < INTER_MODE_CONTEXTS; ++i)
        av1_cost_tokens((int *)cpi->inter_mode_cost[i],
                        cm->fc->inter_mode_probs[i], av1_inter_mode_tree);
#endif  // CONFIG_REF_MV
#if CONFIG_EXT_INTER
      for (i = 0; i < INTER_MODE_CONTEXTS; ++i)
        av1_cost_tokens((int *)cpi->inter_compound_mode_cost[i],
                        cm->fc->inter_compound_mode_probs[i],
                        av1_inter_compound_mode_tree);
      for (i = 0; i < BLOCK_SIZE_GROUPS; ++i)
        av1_cost_tokens((int *)cpi->interintra_mode_cost[i],
                        cm->fc->interintra_mode_prob[i],
                        av1_interintra_mode_tree);
#endif  // CONFIG_EXT_INTER
#if CONFIG_MOTION_VAR || CONFIG_WARPED_MOTION
      for (i = BLOCK_8X8; i < BLOCK_SIZES; i++) {
        av1_cost_tokens((int *)cpi->motion_mode_cost[i],
                        cm->fc->motion_mode_prob[i], av1_motion_mode_tree);
      }
#if CONFIG_MOTION_VAR && CONFIG_WARPED_MOTION
      for (i = BLOCK_8X8; i < BLOCK_SIZES; i++) {
        cpi->motion_mode_cost1[i][0] = av1_cost_bit(cm->fc->obmc_prob[i], 0);
        cpi->motion_mode_cost1[i][1] = av1_cost_bit(cm->fc->obmc_prob[i], 1);
      }
#endif  // CONFIG_MOTION_VAR && CONFIG_WARPED_MOTION
#endif  // CONFIG_MOTION_VAR || CONFIG_WARPED_MOTION
    }
  }
}

  // NOTE: The tables below must be of the same size.

  // The functions described below are sampled at the four most significant
  // bits of x^2 + 8 / 256.

  // Normalized rate:
  // This table models the rate for a Laplacian source with given variance
  // when quantized with a uniform quantizer with given stepsize. The
  // closed form expression is:
  // Rn(x) = H(sqrt(r)) + sqrt(r)*[1 + H(r)/(1 - r)],
  // where r = exp(-sqrt(2) * x) and x = qpstep / sqrt(variance),
  // and H(x) is the binary entropy function.
  static const int H_tab_q10[] = {
    65536,
    6086, 5574, 5275, 5063,  4899, 4764, 4651, 4553,
    4389, 4255, 4142, 4044,  3958, 3881, 3811, 3748,
    3635, 3538, 3453, 3376,  3307, 3244, 3186, 3133,
    3037, 2952, 2877, 2809,  2747, 2690, 2638, 2589,
    2501, 2423, 2353, 2290,  2232, 2179, 2130, 2084,
    2001, 1928, 1862, 1802,  1748, 1698, 1651, 1608,
    1530, 1460, 1398, 1342,  1290, 1243, 1199, 1159,
    1086, 1021,  963,  911,   864,  821,  781,  745,
     680,  623,  574,  530,   490,  455,  424,  395,
     345,  304,  269,  239,   213,  190,  171,  154,
     126,  104,   87,   73,    61,   52,   44,   38,
      28,   21,   16,   12,    10,    8,    6,    5,
       3,    2,    1,    1,     1,    0,    0,
  };
  // Normalized distortion:
  // This table models the normalized distortion for a Laplacian source
  // with given variance when quantized with a uniform quantizer
  // with given stepsize. The closed form expression is:
  // Dn(x) = 1 - 1/sqrt(2) * x / sinh(x/sqrt(2))
  // where x = qpstep / sqrt(variance).
  // Note the actual distortion is Dn * variance.
  static const int dist_tab_q10[] = {
       0,
       0,    1,    1,    1,     2,    2,    2,    3,
       3,    4,    5,    5,     6,    7,    7,    8,
       9,   11,   12,   13,    15,   16,   17,   18,
      21,   24,   26,   29,    31,   34,   36,   39,
      44,   49,   54,   59,    64,   69,   73,   78,
      88,   97,  106,  115,   124,  133,  142,  151,
     167,  184,  200,  215,   231,  245,  260,  274,
     301,  327,  351,  375,   397,  418,  439,  458,
     495,  528,  559,  587,   613,  637,  659,  680,
     717,  749,  777,  801,   823,  842,  859,  874,
     899,  919,  936,  949,   960,  969,  977,  983,
     994, 1001, 1006, 1010,  1013, 1015, 1017, 1018,
    1020, 1022, 1022, 1023,  1023, 1023, 1024,
  };
  static const int xsq_iq_q10[] = {
         0,
         4,      8,     12,     16,      20,     24,     28,     32,
        40,     48,     56,     64,      72,     80,     88,     96,
       112,    128,    144,    160,     176,    192,    208,    224,
       256,    288,    320,    352,     384,    416,    448,    480,
       544,    608,    672,    736,     800,    864,    928,    992,
      1120,   1248,   1376,   1504,    1632,   1760,   1888,   2016,
      2272,   2528,   2784,   3040,    3296,   3552,   3808,   4064,
      4576,   5088,   5600,   6112,    6624,   7136,   7648,   8160,
      9184,  10208,  11232,  12256,   13280,  14304,  15328,  16352,
     18400,  20448,  22496,  24544,   26592,  28640,  30688,  32736,
     36832,  40928,  45024,  49120,   53216,  57312,  61408,  65504,
     73696,  81888,  90080,  98272,  106464, 114656, 122848, 131040,
    147424, 163808, 180192, 196576,  212960, 229344, 245728,
  };

static void model_rd_norm(int xsq_q10, int *r_q10, int *d_q10) {
  const int tmp = (xsq_q10 >> 2) + 8;
  const int k = get_msb(tmp) - 3;
  const int xq = (k << 3) + ((tmp >> k) & 0x7);
  const int one_q10 = 1 << 10;
  const int a_q10 = ((xsq_q10 - xsq_iq_q10[xq]) << 10) >> (2 + k);
  const int b_q10 = one_q10 - a_q10;
  *r_q10 = (H_tab_q10[xq] * b_q10 + H_tab_q10[xq + 1] * a_q10) >> 10;
  *d_q10 = (dist_tab_q10[xq] * b_q10 + dist_tab_q10[xq + 1] * a_q10) >> 10;
}

void av1_model_rd_from_var_lapndz(int64_t var, unsigned int n_log2,
                                  unsigned int qstep, int *rate,
                                  int64_t *dist) {
  // This function models the rate and distortion for a Laplacian
  // source with given variance when quantized with a uniform quantizer
  // with given stepsize. The closed form expressions are in:
  // Hang and Chen, "Source Model for transform video coder and its
  // application - Part I: Fundamental Theory", IEEE Trans. Circ.
  // Sys. for Video Tech., April 1997.
  if (var == 0) {
    *rate = 0;
    *dist = 0;
  } else {
    int d_q10, r_q10;
    static const uint32_t MAX_XSQ_Q10 = 245727;
    const uint64_t xsq_q10_64 =
        (((uint64_t)qstep * qstep << (n_log2 + 10)) + (var >> 1)) / var;
    const int xsq_q10 = (int)AOMMIN(xsq_q10_64, MAX_XSQ_Q10);
    model_rd_norm(xsq_q10, &r_q10, &d_q10);
    *rate = ROUND_POWER_OF_TWO(r_q10 << n_log2, 10 - AV1_PROB_COST_SHIFT);
    *dist = (var * (int64_t)d_q10 + 512) >> 10;
  }
}

#if CONFIG_RD_MODEL
/* the only alpha tables right now are intra for Y plane by txsize */
static const float slope_tab_xorigins[TX_SIZES] =
  { -1.07806f, -.743853f, -.569394f, -.239518f };
static const float slope_tab_yorigins[TX_SIZES] =
  { .0327126f, .0355751f, .0322011f, .0286705f };
static const float slope_tab_scale[TX_SIZES] =
  { 1.f, 1.f, .707107f, .25f };

static const float ialpha_tabs[TX_SIZES][513] = {
  {
    1.7059e+06f, 1.4628e+06f, 1.2803e+06f, 1.1383e+06f, 
    1.0246e+06f, 9.3163e+05f, 6.3948e+05f, 4.0625e+05f, 
    2.6484e+05f, 1.7751e+05f, 1.2008e+05f,      83369.f, 
    58733.f,      42002.f,      30419.f,      22417.f, 
    16750.f,      12610.f,     9652.5f,     7442.6f, 
    5815.3f,     4603.6f,     3650.8f,     2935.4f, 
    2378.6f,     1943.7f,     1604.1f,     1330.2f, 
    1109.3f,     932.19f,     788.04f,     670.14f, 
    573.12f,     492.97f,     426.67f,     370.23f, 
    322.73f,     282.99f,     249.62f,     220.13f, 
    195.36f,     174.37f,     155.46f,     139.56f, 
    125.44f,     113.17f,     102.62f,     93.059f, 
    84.882f,     77.355f,     70.911f,      64.99f, 
    59.854f,     55.171f,     50.979f,     47.273f, 
    43.807f,     40.815f,     37.985f,     35.468f, 
    33.236f,     31.086f,     29.197f,     27.482f, 
    25.844f,     24.391f,     23.071f,     21.781f, 
    20.627f,      19.59f,     18.587f,     17.668f, 
    16.836f,      16.07f,     15.311f,      14.62f, 
    13.989f,     13.405f,     12.829f,     12.301f, 
    11.814f,     11.365f,     10.926f,     10.509f, 
    10.122f,     9.7633f,     9.4289f,     9.1004f, 
    8.7892f,     8.4985f,     8.2265f,     7.9713f, 
    7.7285f,     7.4912f,     7.2681f,     7.0579f, 
    6.8595f,      6.672f,     6.4944f,     6.3224f, 
    6.1592f,     6.0042f,     5.8568f,     5.7165f, 
    5.5827f,     5.4551f,     5.3337f,      5.218f, 
    5.1072f,      5.001f,     4.8992f,     4.8014f, 
    4.7074f,     4.6171f,     4.5301f,     4.4475f, 
    4.3681f,     4.2915f,     4.2176f,     4.1462f, 
    4.0772f,     4.0104f,     3.9457f,     3.8832f, 
    3.8225f,     3.7638f,     3.7083f,     3.6544f, 
    3.6021f,     3.5513f,     3.5019f,     3.4539f, 
    3.4071f,     3.3616f,     3.3173f,     3.2741f, 
    3.2321f,     3.1911f,     3.1512f,     3.1126f, 
    3.0753f,     3.0388f,     3.0031f,     2.9683f, 
    2.9343f,      2.901f,     2.8685f,     2.8367f, 
    2.8057f,     2.7752f,     2.7455f,     2.7163f, 
    2.6878f,     2.6599f,     2.6326f,     2.6053f, 
    2.5783f,     2.5519f,      2.526f,     2.5006f, 
    2.4757f,     2.4513f,     2.4274f,      2.404f, 
     2.381f,     2.3584f,     2.3363f,     2.3145f, 
    2.2932f,     2.2723f,     2.2517f,     2.2315f, 
    2.2117f,     2.1922f,     2.1727f,     2.1535f, 
    2.1346f,     2.1161f,     2.0978f,     2.0799f, 
    2.0623f,      2.045f,     2.0279f,     2.0112f, 
    1.9947f,     1.9785f,     1.9626f,     1.9469f, 
    1.9314f,     1.9162f,     1.9013f,     1.8865f, 
     1.872f,     1.8577f,     1.8437f,     1.8293f, 
      1.81f,      1.791f,     1.7725f,     1.7543f, 
    1.7365f,      1.719f,     1.7019f,     1.6852f, 
    1.6687f,     1.6526f,     1.6368f,     1.6213f, 
     1.606f,     1.5911f,     1.5764f,      1.562f, 
    1.5479f,      1.534f,     1.5115f,     1.4757f, 
    1.4415f,     1.4089f,     1.3777f,     1.3479f, 
    1.3194f,      1.292f,     1.2657f,     1.2552f, 
    1.2454f,     1.2358f,     1.2264f,     1.2171f, 
    1.2079f,     1.1989f,       1.19f,     1.1812f, 
    1.1726f,      1.164f,     1.1557f,     1.1474f, 
    1.1392f,     1.1312f,     1.1233f,     1.1155f, 
    1.1077f,     1.1001f,     1.0926f,     1.0853f, 
     1.078f,     1.0708f,     1.0631f,     1.0553f, 
    1.0477f,     1.0402f,     1.0328f,     1.0255f, 
    1.0183f,     1.0112f,     1.0042f,    0.99731f, 
    0.9905f,    0.98378f,    0.97716f,    0.97062f, 
   0.96417f,    0.95781f,    0.95153f,    0.94533f, 
   0.93921f,    0.93317f,     0.9272f,    0.92132f, 
    0.9155f,    0.90976f,     0.9041f,     0.8985f, 
   0.89297f,    0.88751f,     0.8801f,    0.86393f, 
   0.84835f,    0.83331f,     0.8188f,    0.80479f, 
   0.79124f,    0.77815f,    0.76548f,    0.75322f, 
   0.74134f,    0.73282f,    0.72856f,    0.72434f, 
   0.72018f,    0.71607f,      0.712f,    0.70798f, 
     0.704f,    0.70007f,    0.69618f,    0.69233f, 
   0.68853f,    0.68476f,    0.68104f,    0.67736f, 
   0.67372f,    0.67012f,    0.66655f,    0.66303f, 
   0.65954f,    0.65608f,    0.65267f,    0.64929f, 
   0.64594f,    0.64263f,    0.63935f,     0.6361f, 
   0.63289f,    0.62971f,    0.62656f,    0.62344f, 
   0.62036f,     0.6161f,    0.61168f,    0.60732f, 
   0.60303f,    0.59879f,    0.59461f,    0.59049f, 
   0.58643f,    0.58242f,    0.57846f,    0.57456f, 
   0.57072f,    0.56692f,    0.56317f,    0.55947f, 
   0.55583f,    0.55222f,    0.54867f,    0.54516f, 
   0.54169f,    0.53827f,    0.53489f,    0.53156f, 
   0.52826f,    0.52501f,    0.52179f,    0.51862f, 
   0.51548f,    0.51238f,    0.50918f,    0.50571f, 
   0.50228f,    0.49891f,    0.49557f,    0.49228f, 
   0.48904f,    0.48584f,    0.48268f,    0.47956f, 
   0.47648f,    0.47343f,    0.47043f,    0.46747f, 
   0.46454f,    0.46165f,    0.45879f,    0.45597f, 
   0.45319f,    0.45044f,    0.44772f,    0.44503f, 
   0.44238f,    0.43976f,    0.43717f,     0.4346f, 
   0.43207f,    0.43019f,      0.429f,    0.42781f, 
   0.42663f,    0.42545f,    0.42428f,    0.42312f, 
   0.42197f,    0.42082f,    0.41968f,    0.41854f, 
   0.41741f,    0.41629f,    0.41517f,    0.41406f, 
   0.41295f,    0.41185f,    0.41075f,    0.40967f, 
   0.40858f,    0.40751f,    0.40643f,    0.40537f, 
   0.40431f,    0.40325f,     0.4022f,    0.40116f, 
   0.40012f,    0.39909f,    0.39806f,    0.39704f, 
   0.39602f,    0.39501f,      0.394f,      0.393f, 
     0.392f,    0.39101f,    0.39002f,    0.38904f, 
   0.38807f,    0.38709f,    0.38613f,    0.38516f, 
   0.38421f,    0.38325f,    0.38231f,    0.38136f, 
   0.38042f,    0.37949f,    0.37856f,    0.37764f, 
   0.37672f,     0.3758f,    0.37489f,    0.37398f, 
   0.37308f,    0.37218f,    0.37129f,     0.3704f, 
   0.36951f,    0.36863f,    0.36775f,    0.36688f, 
   0.36601f,    0.36515f,    0.36428f,    0.36343f, 
   0.36258f,    0.36173f,    0.36088f,    0.36004f, 
   0.35921f,    0.35837f,    0.35754f,    0.35672f, 
   0.22513f,    0.22454f,    0.22395f,    0.22337f, 
   0.22279f,    0.22221f,    0.22163f,    0.22106f, 
   0.22049f,    0.21992f,    0.21936f,     0.2188f, 
   0.21824f,    0.21769f,    0.21713f,    0.21658f, 
   0.21604f,    0.21549f,    0.21495f,    0.21441f, 
    0.21388f,    0.21334f,    0.21281f,    0.21229f, 
    0.21176f,    0.21124f,    0.21072f,     0.2102f, 
    0.20968f,    0.20917f,    0.20866f,    0.20815f, 
    0.20765f,    0.20715f,    0.20665f,    0.20615f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f,    0.16967f,    0.16967f,    0.16967f, 
    0.16967f},
  {1.0827e+06f, 1.0124e+06f, 9.5062e+05f, 8.0795e+05f, 
5.6257e+05f, 3.9633e+05f, 2.8507e+05f, 2.0786e+05f, 
1.5457e+05f, 1.1559e+05f,      87278.f,      66672.f, 
     51364.f,      40123.f,      31378.f,      24835.f, 
     19764.f,      15835.f,      12737.f,      10348.f, 
    8460.5f,     6947.3f,     5739.8f,     4775.3f, 
    3979.2f,     3333.5f,     2811.1f,     2377.4f, 
    2021.3f,     1728.6f,     1485.4f,     1284.2f, 
    1109.1f,     962.35f,     839.47f,     736.43f, 
    645.53f,     568.41f,     503.47f,     445.94f, 
    396.18f,     354.36f,     316.42f,      283.9f, 
    255.58f,     230.13f,     208.64f,     188.57f, 
    171.77f,     156.08f,     142.84f,     130.28f, 
    119.69f,     109.65f,     101.17f,     93.068f, 
    86.058f,     79.571f,     73.713f,     68.539f, 
    63.585f,     59.298f,     55.269f,     51.619f, 
    48.422f,     45.254f,     42.462f,     39.963f, 
    37.509f,     35.339f,     33.392f,     31.458f, 
    29.737f,     28.194f,     26.666f,     25.269f, 
    24.011f,     22.831f,     21.699f,     20.674f, 
    19.741f,     18.832f,     17.972f,     17.186f, 
    16.466f,     15.765f,     15.093f,     14.476f, 
    13.907f,     13.372f,     12.839f,     12.347f, 
    11.892f,     11.468f,      11.06f,     10.659f, 
    10.286f,     9.9389f,     9.6141f,     9.3059f, 
    8.9958f,     8.7057f,     8.4337f,     8.1782f, 
    7.9377f,     7.7037f,     7.4717f,     7.2532f, 
    7.0471f,     6.8525f,     6.6682f,     6.4937f, 
    6.3161f,      6.148f,     5.9886f,     5.8372f, 
    5.6933f,     5.5564f,     5.4259f,     5.2971f, 
    5.1737f,      5.056f,     4.9434f,     4.8358f, 
    4.7328f,     4.6341f,     4.5394f,     4.4482f, 
    4.3604f,     4.2761f,      4.195f,     4.1168f, 
    4.0416f,      3.969f,      3.899f,     3.8314f, 
    3.7662f,     3.7036f,     3.6432f,     3.5848f, 
    3.5282f,     3.4733f,     3.4201f,     3.3685f, 
    3.3185f,     3.2699f,     3.2228f,     3.1769f, 
    3.1324f,     3.0896f,     3.0479f,     3.0074f, 
    2.9679f,     2.9294f,     2.8919f,     2.8554f, 
    2.8198f,      2.785f,     2.7511f,      2.718f, 
    2.6857f,     2.6542f,     2.6234f,     2.5918f, 
    2.5601f,     2.5291f,     2.4988f,     2.4693f, 
    2.4405f,     2.4123f,     2.3848f,     2.3579f, 
    2.3316f,     2.3059f,     2.2807f,     2.2561f, 
     2.232f,     2.2085f,     2.1854f,     2.1622f, 
    2.1393f,      2.117f,      2.095f,     2.0736f, 
    2.0525f,     2.0319f,     2.0117f,     1.9919f, 
    1.9725f,     1.9534f,     1.9348f,     1.9164f, 
    1.8985f,     1.8808f,     1.8635f,     1.8465f, 
    1.8298f,     1.8181f,     1.8107f,     1.8035f, 
    1.7963f,     1.7892f,     1.7821f,     1.7751f, 
    1.7681f,     1.7612f,     1.7543f,     1.7475f, 
    1.7408f,     1.7341f,     1.7274f,     1.7208f, 
    1.7143f,     1.7078f,     1.7013f,     1.6949f, 
    1.6885f,     1.6822f,      1.676f,     1.6698f, 
    1.6636f,     1.6575f,     1.6514f,     1.6454f, 
    1.6394f,     1.6334f,     1.6275f,     1.6217f, 
    1.6158f,     1.6101f,     1.6043f,     1.5986f, 
     1.593f,     1.5874f,     1.5818f,     1.5763f, 
    1.5708f,     1.5653f,     1.5599f,     1.5545f, 
    1.5492f,     1.5438f,     1.5386f,     1.5333f, 
    1.5281f,     1.5229f,     1.5161f,     1.5094f, 
    1.5028f,     1.4962f,     1.4897f,     1.4832f, 
    1.4768f,     1.4704f,     1.4641f,     1.4579f, 
    1.4517f,     1.4455f,     1.4394f,     1.4334f, 
    1.4274f,     1.4215f,     1.4156f,     1.4097f, 
    1.4039f,     1.3982f,     1.3925f,     1.3868f, 
    1.3812f,     1.3756f,     1.3701f,     1.3646f, 
    1.3592f,     1.3538f,     1.3485f,     1.3432f, 
    1.3379f,     1.3327f,     1.3275f,     1.3224f, 
    1.3173f,     1.3122f,     1.3072f,     1.3022f, 
    1.2972f,     1.2923f,     1.2875f,     1.2826f, 
    1.2778f,     1.2731f,     1.2646f,     1.2559f, 
    1.2474f,      1.239f,     1.2307f,     1.2225f, 
    1.2144f,     1.2064f,     1.1986f,     1.1908f, 
    1.1831f,     1.1755f,     1.1681f,     1.1607f, 
    1.1534f,     1.1462f,     1.1391f,     1.1321f, 
    1.1252f,     1.1183f,     1.1115f,     1.1049f, 
    1.0982f,     1.0917f,     1.0853f,     1.0789f, 
    1.0726f,     1.0664f,     1.0608f,     1.0555f, 
    1.0503f,     1.0451f,     1.0399f,     1.0348f, 
    1.0298f,     1.0248f,     1.0198f,     1.0149f, 
    1.0101f,     1.0053f,     1.0005f,     0.9958f, 
   0.99113f,     0.9865f,    0.98191f,    0.97737f, 
   0.97287f,    0.96841f,    0.96399f,    0.95961f, 
   0.95527f,    0.95097f,     0.9467f,    0.94248f, 
   0.93829f,    0.93414f,    0.93003f,    0.92595f, 
   0.92191f,    0.91791f,    0.91393f,       0.91f, 
   0.90609f,    0.90222f,    0.89839f,    0.89458f, 
   0.89081f,    0.88331f,    0.87342f,    0.86374f, 
   0.85428f,    0.84502f,    0.83596f,    0.82709f, 
   0.81841f,    0.80991f,    0.80158f,    0.79343f, 
   0.78543f,     0.7776f,    0.76992f,    0.76239f, 
   0.75501f,    0.74777f,     0.7409f,    0.73524f, 
   0.72967f,    0.72419f,    0.71879f,    0.71346f, 
   0.70822f,    0.70305f,    0.69796f,    0.69294f, 
   0.68799f,    0.68311f,     0.6783f,    0.67356f, 
   0.66888f,    0.66427f,    0.65972f,    0.65523f, 
   0.65081f,    0.64644f,    0.64213f,    0.63788f, 
   0.63368f,    0.62954f,    0.62545f,    0.62142f, 
   0.61731f,    0.61182f,    0.60642f,    0.60112f, 
   0.59591f,    0.59079f,    0.58575f,     0.5808f, 
   0.57594f,    0.57115f,    0.56644f,    0.56181f, 
   0.55726f,    0.55278f,    0.54837f,    0.54403f, 
   0.53976f,    0.53555f,    0.53141f,    0.52733f, 
   0.52332f,    0.51937f,    0.51568f,    0.51252f, 
    0.5094f,    0.50631f,    0.50327f,    0.50026f, 
   0.49728f,    0.49434f,    0.49144f,    0.48857f, 
   0.48573f,    0.48292f,    0.48015f,    0.47741f, 
    0.4747f,    0.47202f,    0.46937f,    0.46675f, 
   0.46416f,     0.4616f,    0.45906f,    0.45656f, 
   0.45408f,    0.45162f,     0.4492f,     0.4468f, 
   0.44442f,    0.44207f,    0.43975f,    0.43745f, 
   0.43517f,    0.43292f,    0.43069f,    0.42913f, 
   0.42761f,     0.4261f,     0.4246f,    0.42311f, 
   0.42163f,    0.42016f,     0.4187f,    0.41725f, 
   0.41581f,    0.41439f,    0.41297f,    0.41156f, 
   0.41016f,    0.40877f,    0.40739f,    0.40601f, 
   0.40465f,     0.4033f,    0.40195f,    0.40062f, 
   0.39929f,    0.39797f,    0.39666f,    0.39536f, 
   0.39407f,    0.39279f,    0.39151f,    0.39025f, 
   0.38899f,    0.38774f,    0.38649f,    0.38526f, 
   0.38403f,    0.38281f,     0.3816f,     0.3804f, 
    0.3792f,    0.37801f,    0.37683f,    0.37566f, 
   0.37449f,    0.37333f,    0.37218f,    0.37103f, 
   0.36989f,    0.36876f,    0.36764f,    0.36652f, 
   0.36541f,    0.36431f,    0.36321f,    0.36212f, 
   0.36103f,    0.29894f,    0.29894f,    0.29894f, 
   0.29894f},
  {1.2515e+06f, 1.1649e+06f, 1.0895e+06f, 1.0232e+06f, 
9.6457e+05f, 9.1228e+05f, 5.8934e+05f, 3.8751e+05f, 
2.6348e+05f, 1.8045e+05f, 1.2546e+05f,      88353.f, 
     63987.f,      47203.f,      34844.f,      25797.f, 
     19628.f,      14845.f,      11425.f,     8842.7f, 
    6940.7f,     5543.3f,       4411.f,     3538.1f, 
    2860.6f,       2336.f,     1921.1f,     1585.1f, 
    1314.5f,     1099.1f,     922.73f,      780.6f, 
    662.42f,     564.52f,     484.15f,     418.02f, 
     362.4f,     314.59f,     274.51f,     240.93f, 
    212.12f,     187.08f,     165.96f,      147.6f, 
    131.42f,     117.78f,     105.39f,     94.779f, 
    85.637f,      77.47f,     70.527f,     64.114f, 
    58.677f,     53.606f,     49.274f,     45.223f, 
    41.681f,     38.467f,     35.581f,     33.026f, 
    30.541f,     28.404f,     26.419f,     24.658f, 
      23.1f,     21.615f,      20.31f,     19.116f, 
    17.941f,     16.902f,     15.963f,     15.076f, 
    14.283f,     13.569f,     12.867f,     12.225f, 
    11.644f,     11.106f,     10.582f,     10.105f, 
    9.6698f,     9.2588f,     8.8463f,      8.469f, 
    8.1226f,     7.8034f,     7.4736f,     7.1698f, 
    6.8897f,     6.6307f,     6.3831f,     6.1444f, 
    5.9229f,     5.7168f,     5.5246f,     5.3356f, 
    5.1448f,     4.9672f,     4.8014f,     4.6463f, 
    4.5014f,     4.3738f,     4.2533f,     4.1392f, 
    4.0311f,     3.9285f,     3.8309f,     3.7372f, 
    3.6445f,     3.5562f,     3.4722f,      3.392f, 
    3.3154f,     3.2422f,     3.1722f,      3.108f, 
    3.0513f,     2.9967f,      2.944f,     2.8931f, 
     2.844f,     2.7964f,     2.7505f,      2.706f, 
     2.663f,     2.6213f,     2.5834f,     2.5481f, 
    2.5137f,     2.4803f,     2.4478f,     2.4161f, 
    2.3852f,     2.3551f,     2.3257f,     2.2971f, 
    2.2691f,     2.2418f,     2.2152f,     2.1892f, 
    2.1653f,     2.1447f,     2.1244f,     2.1045f, 
     2.085f,     2.0659f,     2.0471f,     2.0286f, 
    2.0105f,     1.9927f,     1.9752f,      1.958f, 
    1.9411f,     1.9245f,     1.9081f,     1.8921f, 
    1.8763f,     1.8608f,     1.8455f,     1.8305f, 
    1.8157f,      1.802f,     1.7887f,     1.7756f, 
    1.7627f,       1.75f,     1.7375f,     1.7251f, 
     1.713f,     1.7009f,     1.6891f,     1.6774f, 
    1.6659f,     1.6545f,     1.6433f,     1.6323f, 
    1.6214f,     1.6106f,        1.6f,     1.5895f, 
    1.5792f,      1.569f,     1.5589f,     1.5489f, 
    1.5391f,     1.5294f,     1.5198f,     1.5102f, 
    1.5002f,     1.4904f,     1.4807f,     1.4712f, 
    1.4618f,     1.4524f,     1.4432f,     1.4342f, 
    1.4252f,     1.4163f,     1.4076f,      1.399f, 
    1.3904f,      1.382f,     1.3737f,     1.3654f, 
    1.3573f,     1.3493f,     1.3413f,     1.3335f, 
    1.3257f,     1.3181f,     1.3105f,      1.303f, 
    1.2956f,     1.2883f,      1.281f,     1.2739f, 
    1.2668f,     1.2589f,     1.2511f,     1.2434f, 
    1.2358f,     1.2283f,     1.2208f,     1.2135f, 
    1.2063f,     1.1991f,      1.192f,     1.1851f, 
    1.1781f,     1.1713f,     1.1646f,     1.1579f, 
    1.1513f,     1.1448f,     1.1383f,      1.132f, 
    1.1256f,     1.1194f,     1.1132f,     1.1072f, 
    1.1011f,     1.0952f,     1.0892f,     1.0834f, 
    1.0776f,     1.0719f,     1.0663f,     1.0607f, 
    1.0552f,     1.0502f,     1.0451f,     1.0402f, 
    1.0352f,     1.0304f,     1.0255f,     1.0207f, 
     1.016f,     1.0113f,     1.0066f,      1.002f, 
   0.99745f,    0.99292f,    0.98843f,    0.98398f, 
   0.97957f,     0.9752f,    0.97087f,    0.96658f, 
   0.96232f,     0.9581f,    0.95392f,    0.94978f, 
   0.94567f,    0.94159f,    0.93755f,    0.93355f, 
   0.92958f,    0.92564f,    0.92174f,    0.91787f, 
   0.91403f,    0.91022f,    0.90645f,     0.9027f, 
   0.89899f,    0.89531f,    0.89166f,    0.88803f, 
   0.88444f,    0.88088f,    0.87656f,    0.87215f, 
   0.86778f,    0.86346f,    0.85918f,    0.85494f, 
   0.85074f,    0.84659f,    0.84247f,    0.83839f, 
   0.83436f,    0.83036f,     0.8264f,    0.82248f, 
   0.81859f,    0.81474f,    0.81093f,    0.80716f, 
   0.80341f,    0.79971f,    0.79603f,    0.79239f, 
   0.78879f,    0.78521f,    0.78167f,    0.77816f, 
   0.77468f,    0.77124f,    0.76782f,    0.76443f, 
   0.76107f,    0.75775f,    0.75445f,    0.75118f, 
   0.74794f,    0.74472f,    0.74154f,    0.73838f, 
   0.73248f,    0.72188f,    0.71158f,    0.70158f, 
   0.69185f,    0.68239f,    0.67318f,    0.66422f, 
    0.6555f,      0.647f,    0.63872f,    0.63064f, 
   0.62277f,     0.6151f,    0.59791f,    0.58123f, 
   0.56547f,    0.55053f,    0.53637f,    0.52291f, 
   0.50968f,    0.49448f,    0.48016f,    0.46665f, 
   0.45387f,    0.44178f,    0.43031f,    0.35341f, 
   0.35002f,     0.3467f,    0.34344f,    0.34024f, 
    0.3371f,    0.33402f,    0.33099f,    0.32802f, 
    0.3251f,    0.32223f,    0.31941f,    0.31665f, 
   0.31392f,    0.31125f,    0.30862f,    0.30603f, 
   0.30349f,    0.30099f,    0.29853f,    0.29663f, 
   0.29493f,    0.29324f,    0.29157f,    0.28992f, 
   0.28829f,    0.28667f,    0.28508f,     0.2835f, 
   0.28194f,     0.2804f,    0.27887f,    0.27736f, 
   0.27587f,    0.27439f,    0.27293f,    0.27148f, 
   0.27005f,    0.26863f,    0.26723f,    0.26584f, 
   0.26447f,    0.26311f,    0.26177f,    0.26044f, 
   0.25912f,    0.25782f,    0.25652f,    0.25525f, 
   0.25398f,    0.25273f,    0.25149f,    0.25026f, 
   0.24904f,    0.24809f,    0.24736f,    0.24663f, 
    0.2459f,    0.24518f,    0.24447f,    0.24376f, 
   0.24305f,    0.24235f,    0.24165f,    0.24096f, 
   0.24026f,    0.23958f,    0.23889f,    0.23822f, 
   0.23754f,    0.23687f,     0.2362f,    0.23554f, 
   0.23488f,    0.23422f,    0.23357f,    0.23292f, 
   0.23227f,    0.23163f,    0.23099f,    0.23036f, 
   0.22973f,     0.2291f,    0.22848f,    0.22785f, 
   0.22724f,    0.22662f,    0.22601f,     0.2254f, 
    0.2248f,     0.2242f,     0.2236f,      0.223f, 
   0.22241f,    0.22182f,    0.22124f,    0.22066f, 
   0.22008f,     0.2195f,    0.21893f,    0.21836f, 
   0.21779f,    0.21722f,    0.21666f,     0.2161f, 
   0.21555f,      0.215f,    0.21445f,     0.2139f, 
   0.21335f,    0.21281f,    0.21227f,    0.21174f, 
    0.2112f,    0.21067f,    0.21014f,    0.20962f, 
    0.2091f,    0.20857f,    0.20806f,    0.20754f, 
   0.20703f,    0.20667f,     0.2064f,    0.20613f, 
   0.20586f,    0.20559f,    0.20533f,    0.20506f, 
    0.2048f,    0.20453f,    0.20427f,    0.20401f, 
   0.20374f,    0.20348f,    0.20322f,    0.20296f, 
    0.2027f,    0.20244f,    0.20218f,    0.20193f, 
   0.20167f,    0.20141f,    0.20116f,     0.2009f, 
   0.20065f,    0.20039f,    0.20014f,    0.19989f, 
   0.19964f,    0.19939f,    0.19914f,    0.19889f, 
   0.19864f,    0.19839f,    0.19814f,    0.19789f, 
   0.19765f},
  {1.4056e+06f, 1.2993e+06f, 1.2079e+06f, 1.1286e+06f, 
 1.059e+06f, 9.9754e+05f,  9.428e+05f, 7.5264e+05f, 
4.5037e+05f, 2.7118e+05f, 1.6681e+05f, 1.0727e+05f, 
     70502.f,      45725.f,      30913.f,      21084.f, 
     14920.f,      10679.f,     7825.1f,     5666.1f, 
    4179.6f,     3171.6f,     2434.6f,     1888.5f, 
    1496.3f,     1174.7f,      934.5f,     760.07f, 
    611.67f,     505.63f,     414.67f,     344.74f, 
    293.22f,     242.24f,     207.41f,     177.47f, 
    151.29f,     129.59f,     112.55f,      99.13f, 
    86.286f,     75.719f,     66.956f,     59.459f, 
    53.093f,     47.752f,     42.543f,     38.251f, 
    34.648f,     31.339f,     28.461f,     25.804f, 
    23.547f,     21.556f,     19.862f,     18.207f, 
     16.73f,     15.427f,     14.282f,     13.277f, 
    12.343f,     11.533f,     10.743f,     10.011f, 
    9.3715f,     8.8141f,     8.3195f,     7.8774f, 
    7.4256f,     7.0109f,     6.6401f,     6.2936f, 
     5.973f,     5.6835f,     5.4208f,     5.1643f, 
    4.9305f,      4.717f,     4.5212f,     4.3379f, 
    4.1687f,     4.0122f,     3.8671f,     3.7315f, 
    3.6032f,     3.4834f,     3.3713f,     3.2662f, 
    3.1675f,     3.0741f,     2.9859f,     2.9025f, 
    2.8237f,     2.7491f,     2.6783f,      2.611f, 
    2.5543f,     2.5003f,     2.4485f,     2.3988f, 
    2.3511f,     2.3053f,     2.2612f,     2.2188f, 
    2.1779f,     2.1418f,     2.1073f,     2.0738f, 
    2.0414f,       2.01f,     1.9796f,       1.95f, 
    1.9214f,     1.8935f,     1.8665f,     1.8402f, 
    1.8146f,     1.7931f,     1.7728f,      1.753f, 
    1.7336f,     1.7146f,      1.696f,     1.6779f, 
    1.6601f,     1.6427f,     1.6256f,     1.6089f, 
    1.5926f,     1.5766f,     1.5609f,     1.5455f, 
    1.5304f,     1.5156f,     1.5012f,     1.4873f, 
    1.4737f,     1.4603f,     1.4471f,     1.4342f, 
    1.4215f,      1.409f,     1.3967f,     1.3847f, 
    1.3729f,     1.3612f,     1.3498f,     1.3385f, 
    1.3275f,     1.3166f,     1.3059f,     1.2953f, 
     1.285f,     1.2748f,     1.2647f,     1.2551f, 
    1.2458f,     1.2366f,     1.2275f,     1.2186f, 
    1.2098f,     1.2011f,     1.1925f,     1.1841f, 
    1.1758f,     1.1676f,     1.1595f,     1.1515f, 
    1.1436f,     1.1359f,     1.1282f,     1.1206f, 
    1.1132f,     1.1058f,     1.0986f,     1.0914f, 
    1.0843f,     1.0773f,     1.0705f,     1.0637f, 
    1.0569f,     1.0495f,     1.0419f,     1.0344f, 
     1.027f,     1.0197f,     1.0125f,     1.0054f, 
   0.99845f,    0.99155f,    0.98475f,    0.97804f, 
   0.97143f,     0.9649f,    0.95846f,     0.9521f, 
   0.94583f,    0.93964f,    0.93353f,     0.9275f, 
   0.92155f,    0.91567f,    0.90987f,    0.90414f, 
   0.89848f,    0.89289f,    0.88737f,    0.88192f, 
   0.87663f,    0.87177f,    0.86696f,    0.86221f, 
   0.85751f,    0.85286f,    0.84827f,    0.84372f, 
   0.83922f,    0.83476f,    0.83036f,      0.826f, 
   0.82168f,    0.81741f,    0.81319f,      0.809f, 
   0.80487f,    0.80077f,    0.79671f,     0.7927f, 
   0.78872f,    0.78479f,    0.78089f,    0.77704f, 
   0.77322f,    0.76944f,    0.76569f,    0.76198f, 
   0.75831f,    0.75467f,    0.75107f,     0.7475f, 
   0.74396f,    0.74046f,    0.73699f,    0.73356f, 
   0.72921f,    0.72368f,    0.71822f,    0.71285f, 
   0.70756f,    0.70235f,    0.69721f,    0.69215f, 
   0.68716f,    0.68224f,    0.67739f,    0.67261f, 
    0.6679f,    0.66325f,    0.65867f,    0.65415f, 
   0.64969f,    0.64529f,    0.64096f,    0.63667f, 
   0.63245f,    0.62828f,    0.62417f,    0.62011f, 
    0.6161f,    0.61214f,    0.59606f,    0.55878f, 
   0.52588f,    0.48339f,    0.43335f,    0.42036f, 
    0.4151f,    0.40998f,    0.40497f,    0.40009f, 
   0.39533f,    0.39067f,    0.38613f,    0.38169f, 
   0.37735f,    0.37311f,    0.36896f,     0.3649f, 
   0.36093f,    0.35705f,    0.35309f,    0.34069f, 
   0.32913f,    0.31833f,    0.30821f,    0.29872f, 
   0.29426f,    0.29291f,    0.29158f,    0.29025f, 
   0.28894f,    0.28764f,    0.28635f,    0.28507f, 
   0.28381f,    0.28255f,    0.28131f,    0.28008f, 
   0.27885f,    0.27764f,    0.27644f,    0.27525f, 
   0.27407f,     0.2729f,    0.27174f,    0.27059f, 
   0.26945f,    0.26832f,     0.2672f,    0.26608f, 
   0.26498f,    0.26389f,     0.2628f,    0.26172f, 
   0.26066f,     0.2596f,    0.25855f,    0.25751f, 
   0.25647f,    0.25545f,    0.25443f,    0.25342f, 
   0.25242f,    0.25143f,    0.25044f,    0.24947f, 
    0.2485f,    0.24753f,    0.24658f,    0.24563f, 
    0.2436f,    0.24016f,    0.23681f,    0.23355f, 
   0.23039f,     0.2273f,     0.2243f,    0.22138f, 
   0.21853f,    0.21575f,    0.21304f,     0.2104f, 
   0.20783f,    0.20532f,    0.16217f,    0.16119f, 
   0.16022f,    0.15927f,    0.15832f,    0.15738f, 
   0.15646f,    0.15555f,    0.15464f,    0.15375f, 
   0.15287f,      0.152f,    0.15113f,    0.15028f, 
   0.14944f,    0.14861f,    0.14778f,    0.14697f, 
   0.14616f,    0.14536f,    0.14457f,    0.14379f, 
   0.14312f,    0.14286f,    0.14261f,    0.14235f, 
    0.1421f,    0.14185f,     0.1416f,    0.14135f, 
    0.1411f,    0.14085f,     0.1406f,    0.14035f, 
   0.14011f,    0.13986f,    0.13962f,    0.13938f, 
   0.13913f,    0.13889f,    0.13865f,    0.13841f, 
   0.13817f,    0.13793f,     0.1377f,    0.13746f, 
   0.13722f,    0.13699f,    0.13675f,    0.13652f, 
   0.13629f,    0.13606f,    0.13582f,    0.13559f, 
   0.13536f,    0.13514f,    0.13491f,    0.13468f, 
   0.13445f,    0.13423f,      0.134f,    0.13378f, 
   0.13356f,    0.13333f,    0.13311f,    0.13289f, 
   0.13267f,    0.13245f,    0.13223f,    0.13201f, 
   0.13179f,    0.13158f,    0.13136f,    0.13115f, 
   0.13093f,    0.13072f,     0.1305f,    0.13029f, 
   0.13008f,    0.12987f,    0.12966f,    0.12945f, 
   0.12924f,    0.12903f,    0.12882f,    0.12861f, 
   0.12841f,     0.1282f,      0.128f,    0.12779f, 
   0.12759f,    0.12738f,    0.12718f,    0.12698f, 
   0.12678f,    0.12658f,    0.12638f,    0.12618f, 
   0.12598f,    0.12578f,    0.12558f,    0.12539f, 
   0.12519f,    0.12499f,     0.1248f,    0.12461f, 
   0.12441f,    0.12422f,    0.12403f,    0.12383f, 
   0.12364f,    0.12345f,    0.12326f,    0.12307f, 
   0.12288f,    0.12269f,    0.12251f,    0.12232f, 
   0.12213f,    0.12194f,    0.12176f,    0.12157f, 
   0.12139f,    0.12121f,    0.12102f,    0.12084f, 
   0.12066f,    0.12047f,    0.12029f,    0.12011f, 
   0.11993f,    0.11975f,    0.11957f,     0.1194f, 
   0.11641f,   0.059953f,   0.040373f,   0.038833f, 
  0.038833f,   0.038833f,   0.038833f,   0.038833f, 
  0.038833f,   0.038833f,   0.038833f,   0.038833f, 
  0.038833f,   0.038833f,   0.038833f,   0.038833f, 
  0.038833f,   0.038833f,   0.038833f,   0.038833f, 
  0.038833f,   0.038833f,   0.038833f,   0.038833f, 
   0.038833f}
};

float p_laplacian(float std_dev){
  return expf(-sqrtf(2.f) / std_dev);
}

float H2_laplacian(float p){
  return p*log2f(1.f/p) + (1.f - p)*log2f(1.f/(1.f-p));
}

float H_laplacian(float std_dev){
  float p = p_laplacian(std_dev);
  float p2 = sqrtf(p);
  return H2_laplacian(p2) + p2*(1.f+H2_laplacian(p)/(1.f-p));
}

/* variance and satd have _not_ been divided by txsize */
/* qstep is _unshifted_ */
void av1_model_rate_from_var_satd_lapndz(int64_t var, int satd,
                                         TX_SIZE tx_size,
                                         unsigned int qstep,
                                         unsigned int qstep_unshifted,
                                         int *rate){
  if (var == 0 || satd == 0) {
    *rate = 0;
  } else {
    /* don't get fancy yet. */
    const int n_log2 = tx_size_wide_log2[tx_size] + tx_size_high_log2[tx_size];
    const int n = tx_size_2d[tx_size];
    const float stddev = sqrtf(var/(float)n);
    const float Hval = H_laplacian(stddev/qstep_unshifted);
    const float xorigin = slope_tab_xorigins[tx_size];
    const float yorigin = slope_tab_yorigins[tx_size];
    const float slope = (stddev/qstep_unshifted-yorigin)/
      ((float)satd/n/qstep_unshifted-xorigin);
    const int slope_q16 = AOMMAX(0,
      AOMMIN((int)rint(slope * slope_tab_scale[tx_size]*(1 << 16)),
             (int)(1<<15)-1));
    //fprintf(stderr,"slope:%f slope_q10:%d\n",slope,slope_q16);
    const int slope_i = slope_q16 >> 6;
    const float slope_f = (slope_q16 & 0x3f)*.015625f;
    const float *ialpha_tab = ialpha_tabs[tx_size];
    const float ialpha = (ialpha_tab[slope_i] * (1.f-slope_f) +
                          ialpha_tab[slope_i + 1] * slope_f);

    *rate = ROUND_POWER_OF_TWO((int)rint((Hval * (1<<(n_log2+10)))*ialpha),
                               10 - AV1_PROB_COST_SHIFT);
    //fprintf(stderr,"var:%ld stddev:%f satd:%f H:%f slope:%f slopei:%d slope_f:%.3f ialpha:%f rate:%f\n",
    //        var, stddev/qstep_unshifted, (float)satd/n/qstep_unshifted,
    //      Hval,slope,slope_i,slope_f,ialpha,*rate/(float)(1<<AV1_PROB_COST_SHIFT));
  }
}
#endif
static void get_entropy_contexts_plane(
    BLOCK_SIZE plane_bsize, TX_SIZE tx_size, const struct macroblockd_plane *pd,
    ENTROPY_CONTEXT t_above[2 * MAX_MIB_SIZE],
    ENTROPY_CONTEXT t_left[2 * MAX_MIB_SIZE]) {
  const int num_4x4_w = block_size_wide[plane_bsize] >> tx_size_wide_log2[0];
  const int num_4x4_h = block_size_high[plane_bsize] >> tx_size_high_log2[0];
  const ENTROPY_CONTEXT *const above = pd->above_context;
  const ENTROPY_CONTEXT *const left = pd->left_context;

  int i;

#if CONFIG_CB4X4
  switch (tx_size) {
    case TX_2X2:
      memcpy(t_above, above, sizeof(ENTROPY_CONTEXT) * num_4x4_w);
      memcpy(t_left, left, sizeof(ENTROPY_CONTEXT) * num_4x4_h);
      break;
    case TX_4X4:
      for (i = 0; i < num_4x4_w; i += 2)
        t_above[i] = !!*(const uint16_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 2)
        t_left[i] = !!*(const uint16_t *)&left[i];
      break;
    case TX_8X8:
      for (i = 0; i < num_4x4_w; i += 4)
        t_above[i] = !!*(const uint32_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 4)
        t_left[i] = !!*(const uint32_t *)&left[i];
      break;
    case TX_16X16:
      for (i = 0; i < num_4x4_w; i += 8)
        t_above[i] = !!*(const uint64_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 8)
        t_left[i] = !!*(const uint64_t *)&left[i];
      break;
    case TX_32X32:
      for (i = 0; i < num_4x4_w; i += 16)
        t_above[i] =
            !!(*(const uint64_t *)&above[i] | *(const uint64_t *)&above[i + 8]);
      for (i = 0; i < num_4x4_h; i += 16)
        t_left[i] =
            !!(*(const uint64_t *)&left[i] | *(const uint64_t *)&left[i + 8]);
      break;
    case TX_4X8:
      for (i = 0; i < num_4x4_w; i += 2)
        t_above[i] = !!*(const uint16_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 4)
        t_left[i] = !!*(const uint32_t *)&left[i];
      break;
    case TX_8X4:
      for (i = 0; i < num_4x4_w; i += 4)
        t_above[i] = !!*(const uint32_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 2)
        t_left[i] = !!*(const uint16_t *)&left[i];
      break;
    case TX_8X16:
      for (i = 0; i < num_4x4_w; i += 4)
        t_above[i] = !!*(const uint32_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 8)
        t_left[i] = !!*(const uint64_t *)&left[i];
      break;
    case TX_16X8:
      for (i = 0; i < num_4x4_w; i += 8)
        t_above[i] = !!*(const uint64_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 4)
        t_left[i] = !!*(const uint32_t *)&left[i];
      break;
    case TX_16X32:
      for (i = 0; i < num_4x4_w; i += 8)
        t_above[i] = !!*(const uint64_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 16)
        t_left[i] =
            !!(*(const uint64_t *)&left[i] | *(const uint64_t *)&left[i + 8]);
      break;
    case TX_32X16:
      for (i = 0; i < num_4x4_w; i += 16)
        t_above[i] =
            !!(*(const uint64_t *)&above[i] | *(const uint64_t *)&above[i + 8]);
      for (i = 0; i < num_4x4_h; i += 8)
        t_left[i] = !!*(const uint64_t *)&left[i];
      break;

    default: assert(0 && "Invalid transform size."); break;
  }
  return;
#endif

  switch (tx_size) {
    case TX_4X4:
      memcpy(t_above, above, sizeof(ENTROPY_CONTEXT) * num_4x4_w);
      memcpy(t_left, left, sizeof(ENTROPY_CONTEXT) * num_4x4_h);
      break;
    case TX_8X8:
      for (i = 0; i < num_4x4_w; i += 2)
        t_above[i] = !!*(const uint16_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 2)
        t_left[i] = !!*(const uint16_t *)&left[i];
      break;
    case TX_16X16:
      for (i = 0; i < num_4x4_w; i += 4)
        t_above[i] = !!*(const uint32_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 4)
        t_left[i] = !!*(const uint32_t *)&left[i];
      break;
    case TX_32X32:
      for (i = 0; i < num_4x4_w; i += 8)
        t_above[i] = !!*(const uint64_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 8)
        t_left[i] = !!*(const uint64_t *)&left[i];
      break;
#if CONFIG_TX64X64
    case TX_64X64:
      for (i = 0; i < num_4x4_w; i += 16)
        t_above[i] =
            !!(*(const uint64_t *)&above[i] | *(const uint64_t *)&above[i + 8]);
      for (i = 0; i < num_4x4_h; i += 16)
        t_left[i] =
            !!(*(const uint64_t *)&left[i] | *(const uint64_t *)&left[i + 8]);
      break;
#endif  // CONFIG_TX64X64
    case TX_4X8:
      memcpy(t_above, above, sizeof(ENTROPY_CONTEXT) * num_4x4_w);
      for (i = 0; i < num_4x4_h; i += 2)
        t_left[i] = !!*(const uint16_t *)&left[i];
      break;
    case TX_8X4:
      for (i = 0; i < num_4x4_w; i += 2)
        t_above[i] = !!*(const uint16_t *)&above[i];
      memcpy(t_left, left, sizeof(ENTROPY_CONTEXT) * num_4x4_h);
      break;
    case TX_8X16:
      for (i = 0; i < num_4x4_w; i += 2)
        t_above[i] = !!*(const uint16_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 4)
        t_left[i] = !!*(const uint32_t *)&left[i];
      break;
    case TX_16X8:
      for (i = 0; i < num_4x4_w; i += 4)
        t_above[i] = !!*(const uint32_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 2)
        t_left[i] = !!*(const uint16_t *)&left[i];
      break;
    case TX_16X32:
      for (i = 0; i < num_4x4_w; i += 4)
        t_above[i] = !!*(const uint32_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 8)
        t_left[i] = !!*(const uint64_t *)&left[i];
      break;
    case TX_32X16:
      for (i = 0; i < num_4x4_w; i += 8)
        t_above[i] = !!*(const uint64_t *)&above[i];
      for (i = 0; i < num_4x4_h; i += 4)
        t_left[i] = !!*(const uint32_t *)&left[i];
      break;
    default: assert(0 && "Invalid transform size."); break;
  }
}

void av1_get_entropy_contexts(BLOCK_SIZE bsize, TX_SIZE tx_size,
                              const struct macroblockd_plane *pd,
                              ENTROPY_CONTEXT t_above[2 * MAX_MIB_SIZE],
                              ENTROPY_CONTEXT t_left[2 * MAX_MIB_SIZE]) {
  const BLOCK_SIZE plane_bsize = get_plane_block_size(bsize, pd);
  get_entropy_contexts_plane(plane_bsize, tx_size, pd, t_above, t_left);
}

void av1_mv_pred(const AV1_COMP *cpi, MACROBLOCK *x, uint8_t *ref_y_buffer,
                 int ref_y_stride, int ref_frame, BLOCK_SIZE block_size) {
  int i;
  int zero_seen = 0;
  int best_index = 0;
  int best_sad = INT_MAX;
  int this_sad = INT_MAX;
  int max_mv = 0;
  int near_same_nearest;
  uint8_t *src_y_ptr = x->plane[0].src.buf;
  uint8_t *ref_y_ptr;
  const int num_mv_refs =
      MAX_MV_REF_CANDIDATES +
      (cpi->sf.adaptive_motion_search && block_size < x->max_partition_size);

  MV pred_mv[3];
  pred_mv[0] = x->mbmi_ext->ref_mvs[ref_frame][0].as_mv;
  pred_mv[1] = x->mbmi_ext->ref_mvs[ref_frame][1].as_mv;
  pred_mv[2] = x->pred_mv[ref_frame];
  assert(num_mv_refs <= (int)(sizeof(pred_mv) / sizeof(pred_mv[0])));

  near_same_nearest = x->mbmi_ext->ref_mvs[ref_frame][0].as_int ==
                      x->mbmi_ext->ref_mvs[ref_frame][1].as_int;
  // Get the sad for each candidate reference mv.
  for (i = 0; i < num_mv_refs; ++i) {
    const MV *this_mv = &pred_mv[i];
    int fp_row, fp_col;

    if (i == 1 && near_same_nearest) continue;
    fp_row = (this_mv->row + 3 + (this_mv->row >= 0)) >> 3;
    fp_col = (this_mv->col + 3 + (this_mv->col >= 0)) >> 3;
    max_mv = AOMMAX(max_mv, AOMMAX(abs(this_mv->row), abs(this_mv->col)) >> 3);

    if (fp_row == 0 && fp_col == 0 && zero_seen) continue;
    zero_seen |= (fp_row == 0 && fp_col == 0);

    ref_y_ptr = &ref_y_buffer[ref_y_stride * fp_row + fp_col];
    // Find sad for current vector.
    this_sad = cpi->fn_ptr[block_size].sdf(src_y_ptr, x->plane[0].src.stride,
                                           ref_y_ptr, ref_y_stride);
    // Note if it is the best so far.
    if (this_sad < best_sad) {
      best_sad = this_sad;
      best_index = i;
    }
  }

  // Note the index of the mv that worked best in the reference list.
  x->mv_best_ref_index[ref_frame] = best_index;
  x->max_mv_context[ref_frame] = max_mv;
  x->pred_mv_sad[ref_frame] = best_sad;
}

void av1_setup_pred_block(const MACROBLOCKD *xd,
                          struct buf_2d dst[MAX_MB_PLANE],
                          const YV12_BUFFER_CONFIG *src, int mi_row, int mi_col,
                          const struct scale_factors *scale,
                          const struct scale_factors *scale_uv) {
  int i;

  dst[0].buf = src->y_buffer;
  dst[0].stride = src->y_stride;
  dst[1].buf = src->u_buffer;
  dst[2].buf = src->v_buffer;
  dst[1].stride = dst[2].stride = src->uv_stride;

  for (i = 0; i < MAX_MB_PLANE; ++i) {
    setup_pred_plane(dst + i, dst[i].buf,
                     i ? src->uv_crop_width : src->y_crop_width,
                     i ? src->uv_crop_height : src->y_crop_height,
                     dst[i].stride, mi_row, mi_col, i ? scale_uv : scale,
                     xd->plane[i].subsampling_x, xd->plane[i].subsampling_y);
  }
}

int av1_raster_block_offset(BLOCK_SIZE plane_bsize, int raster_block,
                            int stride) {
  const int bw = b_width_log2_lookup[plane_bsize];
  const int y = 4 * (raster_block >> bw);
  const int x = 4 * (raster_block & ((1 << bw) - 1));
  return y * stride + x;
}

int16_t *av1_raster_block_offset_int16(BLOCK_SIZE plane_bsize, int raster_block,
                                       int16_t *base) {
  const int stride = block_size_wide[plane_bsize];
  return base + av1_raster_block_offset(plane_bsize, raster_block, stride);
}

YV12_BUFFER_CONFIG *av1_get_scaled_ref_frame(const AV1_COMP *cpi,
                                             int ref_frame) {
  const AV1_COMMON *const cm = &cpi->common;
  const int scaled_idx = cpi->scaled_ref_idx[ref_frame - 1];
  const int ref_idx = get_ref_frame_buf_idx(cpi, ref_frame);
  return (scaled_idx != ref_idx && scaled_idx != INVALID_IDX)
             ? &cm->buffer_pool->frame_bufs[scaled_idx].buf
             : NULL;
}

#if CONFIG_DUAL_FILTER
int av1_get_switchable_rate(const AV1_COMP *cpi, const MACROBLOCKD *xd) {
  const AV1_COMMON *const cm = &cpi->common;
  if (cm->interp_filter == SWITCHABLE) {
    const MB_MODE_INFO *const mbmi = &xd->mi[0]->mbmi;
    int inter_filter_cost = 0;
    int dir;

    for (dir = 0; dir < 2; ++dir) {
      if (has_subpel_mv_component(xd->mi[0], xd, dir) ||
          (mbmi->ref_frame[1] > INTRA_FRAME &&
           has_subpel_mv_component(xd->mi[0], xd, dir + 2))) {
        const int ctx = av1_get_pred_context_switchable_interp(xd, dir);
        inter_filter_cost +=
            cpi->switchable_interp_costs[ctx][mbmi->interp_filter[dir]];
      }
    }
    return SWITCHABLE_INTERP_RATE_FACTOR * inter_filter_cost;
  } else {
    return 0;
  }
}
#else
int av1_get_switchable_rate(const AV1_COMP *cpi, const MACROBLOCKD *xd) {
  const AV1_COMMON *const cm = &cpi->common;
  if (cm->interp_filter == SWITCHABLE) {
    const MB_MODE_INFO *const mbmi = &xd->mi[0]->mbmi;
    const int ctx = av1_get_pred_context_switchable_interp(xd);
    return SWITCHABLE_INTERP_RATE_FACTOR *
           cpi->switchable_interp_costs[ctx][mbmi->interp_filter];
  }
  return 0;
}
#endif

void av1_set_rd_speed_thresholds(AV1_COMP *cpi) {
  int i;
  RD_OPT *const rd = &cpi->rd;
  SPEED_FEATURES *const sf = &cpi->sf;

  // Set baseline threshold values.
  for (i = 0; i < MAX_MODES; ++i)
    rd->thresh_mult[i] = cpi->oxcf.mode == BEST ? -500 : 0;

  if (sf->adaptive_rd_thresh) {
    rd->thresh_mult[THR_NEARESTMV] = 300;
#if CONFIG_EXT_REFS
    rd->thresh_mult[THR_NEARESTL2] = 300;
    rd->thresh_mult[THR_NEARESTL3] = 300;
    rd->thresh_mult[THR_NEARESTB] = 300;
#endif  // CONFIG_EXT_REFS
    rd->thresh_mult[THR_NEARESTA] = 300;
    rd->thresh_mult[THR_NEARESTG] = 300;
  } else {
    rd->thresh_mult[THR_NEARESTMV] = 0;
#if CONFIG_EXT_REFS
    rd->thresh_mult[THR_NEARESTL2] = 0;
    rd->thresh_mult[THR_NEARESTL3] = 0;
    rd->thresh_mult[THR_NEARESTB] = 0;
#endif  // CONFIG_EXT_REFS
    rd->thresh_mult[THR_NEARESTA] = 0;
    rd->thresh_mult[THR_NEARESTG] = 0;
  }

  rd->thresh_mult[THR_DC] += 1000;

  rd->thresh_mult[THR_NEWMV] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_NEWL2] += 1000;
  rd->thresh_mult[THR_NEWL3] += 1000;
  rd->thresh_mult[THR_NEWB] += 1000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_NEWA] += 1000;
  rd->thresh_mult[THR_NEWG] += 1000;

  rd->thresh_mult[THR_NEARMV] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_NEARL2] += 1000;
  rd->thresh_mult[THR_NEARL3] += 1000;
  rd->thresh_mult[THR_NEARB] += 1000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_NEARA] += 1000;
  rd->thresh_mult[THR_NEARG] += 1000;

#if CONFIG_EXT_INTER
  rd->thresh_mult[THR_NEWFROMNEARMV] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_NEWFROMNEARL2] += 1000;
  rd->thresh_mult[THR_NEWFROMNEARL3] += 1000;
  rd->thresh_mult[THR_NEWFROMNEARB] += 1000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_NEWFROMNEARA] += 1000;
  rd->thresh_mult[THR_NEWFROMNEARG] += 1000;
#endif  // CONFIG_EXT_INTER

  rd->thresh_mult[THR_ZEROMV] += 2000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_ZEROL2] += 2000;
  rd->thresh_mult[THR_ZEROL3] += 2000;
  rd->thresh_mult[THR_ZEROB] += 2000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_ZEROG] += 2000;
  rd->thresh_mult[THR_ZEROA] += 2000;

  rd->thresh_mult[THR_TM] += 1000;

#if CONFIG_EXT_INTER

  rd->thresh_mult[THR_COMP_NEAREST_NEARESTLA] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTL2A] += 1000;
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTL3A] += 1000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTGA] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTLB] += 1000;
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTL2B] += 1000;
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTL3B] += 1000;
  rd->thresh_mult[THR_COMP_NEAREST_NEARESTGB] += 1000;
#endif  // CONFIG_EXT_REFS

#else  // CONFIG_EXT_INTER

  rd->thresh_mult[THR_COMP_NEARESTLA] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEARESTL2A] += 1000;
  rd->thresh_mult[THR_COMP_NEARESTL3A] += 1000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEARESTGA] += 1000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEARESTLB] += 1000;
  rd->thresh_mult[THR_COMP_NEARESTL2B] += 1000;
  rd->thresh_mult[THR_COMP_NEARESTL3B] += 1000;
  rd->thresh_mult[THR_COMP_NEARESTGB] += 1000;
#endif  // CONFIG_EXT_REFS

#endif  // CONFIG_EXT_INTER

#if CONFIG_EXT_INTER

  rd->thresh_mult[THR_COMP_NEAREST_NEARLA] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTLA] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARLA] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWLA] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTLA] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWLA] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARLA] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWLA] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROLA] += 2500;

#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEAREST_NEARL2A] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTL2A] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARL2A] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWL2A] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTL2A] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWL2A] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARL2A] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWL2A] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROL2A] += 2500;

  rd->thresh_mult[THR_COMP_NEAREST_NEARL3A] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTL3A] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARL3A] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWL3A] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTL3A] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWL3A] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARL3A] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWL3A] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROL3A] += 2500;
#endif  // CONFIG_EXT_REFS

  rd->thresh_mult[THR_COMP_NEAREST_NEARGA] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTGA] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARGA] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWGA] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTGA] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWGA] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARGA] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWGA] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROGA] += 2500;

#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEAREST_NEARLB] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTLB] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARLB] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWLB] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTLB] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWLB] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARLB] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWLB] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROLB] += 2500;

  rd->thresh_mult[THR_COMP_NEAREST_NEARL2B] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTL2B] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARL2B] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWL2B] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTL2B] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWL2B] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARL2B] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWL2B] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROL2B] += 2500;

  rd->thresh_mult[THR_COMP_NEAREST_NEARL3B] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTL3B] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARL3B] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWL3B] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTL3B] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWL3B] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARL3B] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWL3B] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROL3B] += 2500;

  rd->thresh_mult[THR_COMP_NEAREST_NEARGB] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARESTGB] += 1200;
  rd->thresh_mult[THR_COMP_NEAR_NEARGB] += 1200;
  rd->thresh_mult[THR_COMP_NEAREST_NEWGB] += 1500;
  rd->thresh_mult[THR_COMP_NEW_NEARESTGB] += 1500;
  rd->thresh_mult[THR_COMP_NEAR_NEWGB] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEARGB] += 1700;
  rd->thresh_mult[THR_COMP_NEW_NEWGB] += 2000;
  rd->thresh_mult[THR_COMP_ZERO_ZEROGB] += 2500;
#endif  // CONFIG_EXT_REFS

#else  // CONFIG_EXT_INTER

  rd->thresh_mult[THR_COMP_NEARLA] += 1500;
  rd->thresh_mult[THR_COMP_NEWLA] += 2000;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEARL2A] += 1500;
  rd->thresh_mult[THR_COMP_NEWL2A] += 2000;
  rd->thresh_mult[THR_COMP_NEARL3A] += 1500;
  rd->thresh_mult[THR_COMP_NEWL3A] += 2000;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEARGA] += 1500;
  rd->thresh_mult[THR_COMP_NEWGA] += 2000;

#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_NEARLB] += 1500;
  rd->thresh_mult[THR_COMP_NEWLB] += 2000;
  rd->thresh_mult[THR_COMP_NEARL2B] += 1500;
  rd->thresh_mult[THR_COMP_NEWL2B] += 2000;
  rd->thresh_mult[THR_COMP_NEARL3B] += 1500;
  rd->thresh_mult[THR_COMP_NEWL3B] += 2000;
  rd->thresh_mult[THR_COMP_NEARGB] += 1500;
  rd->thresh_mult[THR_COMP_NEWGB] += 2000;
#endif  // CONFIG_EXT_REFS

  rd->thresh_mult[THR_COMP_ZEROLA] += 2500;
#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_ZEROL2A] += 2500;
  rd->thresh_mult[THR_COMP_ZEROL3A] += 2500;
#endif  // CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_ZEROGA] += 2500;

#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_ZEROLB] += 2500;
  rd->thresh_mult[THR_COMP_ZEROL2B] += 2500;
  rd->thresh_mult[THR_COMP_ZEROL3B] += 2500;
  rd->thresh_mult[THR_COMP_ZEROGB] += 2500;
#endif  // CONFIG_EXT_REFS

#endif  // CONFIG_EXT_INTER

  rd->thresh_mult[THR_H_PRED] += 2000;
  rd->thresh_mult[THR_V_PRED] += 2000;
  rd->thresh_mult[THR_D135_PRED] += 2500;
  rd->thresh_mult[THR_D207_PRED] += 2500;
  rd->thresh_mult[THR_D153_PRED] += 2500;
  rd->thresh_mult[THR_D63_PRED] += 2500;
  rd->thresh_mult[THR_D117_PRED] += 2500;
  rd->thresh_mult[THR_D45_PRED] += 2500;

#if CONFIG_EXT_INTER
  rd->thresh_mult[THR_COMP_INTERINTRA_ZEROL] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARESTL] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARL] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEWL] += 2000;

#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_INTERINTRA_ZEROL2] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARESTL2] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARL2] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEWL2] += 2000;

  rd->thresh_mult[THR_COMP_INTERINTRA_ZEROL3] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARESTL3] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARL3] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEWL3] += 2000;
#endif  // CONFIG_EXT_REFS

  rd->thresh_mult[THR_COMP_INTERINTRA_ZEROG] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARESTG] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARG] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEWG] += 2000;

#if CONFIG_EXT_REFS
  rd->thresh_mult[THR_COMP_INTERINTRA_ZEROB] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARESTB] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARB] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEWB] += 2000;
#endif  // CONFIG_EXT_REFS

  rd->thresh_mult[THR_COMP_INTERINTRA_ZEROA] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARESTA] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEARA] += 1500;
  rd->thresh_mult[THR_COMP_INTERINTRA_NEWA] += 2000;
#endif  // CONFIG_EXT_INTER
}

void av1_set_rd_speed_thresholds_sub8x8(AV1_COMP *cpi) {
  static const int thresh_mult[2][MAX_REFS] = {
#if CONFIG_EXT_REFS
    { 2500, 2500, 2500, 2500, 2500, 2500, 4500, 4500, 4500, 4500, 4500, 4500,
      4500, 4500, 2500 },
    { 2000, 2000, 2000, 2000, 2000, 2000, 4000, 4000, 4000, 4000, 4000, 4000,
      4000, 4000, 2000 }
#else
    { 2500, 2500, 2500, 4500, 4500, 2500 },
    { 2000, 2000, 2000, 4000, 4000, 2000 }
#endif  // CONFIG_EXT_REFS
  };
  RD_OPT *const rd = &cpi->rd;
  const int idx = cpi->oxcf.mode == BEST;
  memcpy(rd->thresh_mult_sub8x8, thresh_mult[idx], sizeof(thresh_mult[idx]));
}

void av1_update_rd_thresh_fact(const AV1_COMMON *const cm,
                               int (*factor_buf)[MAX_MODES], int rd_thresh,
                               int bsize, int best_mode_index) {
  if (rd_thresh > 0) {
#if CONFIG_CB4X4
    const int top_mode = MAX_MODES;
#else
    const int top_mode = bsize < BLOCK_8X8 ? MAX_REFS : MAX_MODES;
#endif
    int mode;
    for (mode = 0; mode < top_mode; ++mode) {
      const BLOCK_SIZE min_size = AOMMAX(bsize - 1, BLOCK_4X4);
      const BLOCK_SIZE max_size = AOMMIN(bsize + 2, (int)cm->sb_size);
      BLOCK_SIZE bs;
      for (bs = min_size; bs <= max_size; ++bs) {
        int *const fact = &factor_buf[bs][mode];
        if (mode == best_mode_index) {
          *fact -= (*fact >> 4);
        } else {
          *fact = AOMMIN(*fact + RD_THRESH_INC, rd_thresh * RD_THRESH_MAX_FACT);
        }
      }
    }
  }
}

int av1_get_intra_cost_penalty(int qindex, int qdelta,
                               aom_bit_depth_t bit_depth) {
  const int q = av1_dc_quant(qindex, qdelta, bit_depth);
#if CONFIG_AOM_HIGHBITDEPTH
  switch (bit_depth) {
    case AOM_BITS_8: return 20 * q;
    case AOM_BITS_10: return 5 * q;
    case AOM_BITS_12: return ROUND_POWER_OF_TWO(5 * q, 2);
    default:
      assert(0 && "bit_depth should be AOM_BITS_8, AOM_BITS_10 or AOM_BITS_12");
      return -1;
  }
#else
  return 20 * q;
#endif  // CONFIG_AOM_HIGHBITDEPTH
}
