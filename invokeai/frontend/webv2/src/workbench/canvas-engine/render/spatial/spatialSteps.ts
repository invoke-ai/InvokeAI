// SPDX-License-Identifier: Apache-2.0
export type SpatialAdjustmentStep =
  | { readonly kind: 'clarity'; readonly amount: number; readonly size: number }
  | { readonly kind: 'texture'; readonly amount: number; readonly size: number }
  | { readonly kind: 'dehaze'; readonly amount: number };

/** Largest alpha-bounds area one evaluation accepts; bounds worker scratch memory. */
export const MAX_SPATIAL_PIXELS = 4096 * 4096;

/**
 * Worst-case scratch one evaluation allocates per alpha-bounds pixel: five float32 planes (lightness, adjusted
 * lightness and three filter planes) plus int32 fill labels when the bounds contain hidden pixels.
 */
export const SPATIAL_SCRATCH_BYTES_PER_PIXEL = 24;

export type SpatialEvaluation = { readonly status: 'ok' } | { readonly status: 'too-large' };
