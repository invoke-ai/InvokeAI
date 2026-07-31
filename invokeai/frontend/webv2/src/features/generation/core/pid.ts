import type { KnownGenerationModelBase as KnownModelBase } from '@features/generation/core/contracts';

import type { PidMode } from './types';

/**
 * PiD (Pixel Diffusion Decoder) — NVIDIA's few-step pixel-diffusion decoder that
 * replaces the VAE decode with a caption-conditioned 4x super-resolution decode.
 *
 * Pure geometry and base-compatibility rules; the graph wiring lives in `graph.ts`.
 */

/** PiD's fixed super-resolution factor. Every released checkpoint is 4x. */
export const PID_SCALE = 4;

/** Default decode steps. The released checkpoints are 4-step distillations. */
export const DEFAULT_PID_STEPS = 4;
export const MIN_PID_STEPS = 1;
export const MAX_PID_STEPS = 16;

/**
 * PiD res2k decoders are trained 512 -> 2048. In native mode the user-facing
 * dimensions are the 4x target, so the optimal *target* side is 2048 regardless of the
 * main model's own optimum.
 */
const PID_NATIVE_OPTIMAL_SIDE = 512 * PID_SCALE;

export const PID_MODES: readonly PidMode[] = ['off', 'fit', 'native'];

export const isPidMode = (value: unknown): value is PidMode =>
  typeof value === 'string' && (PID_MODES as readonly string[]).includes(value);

/**
 * The generation scale the dimension helpers must account for: 4 in native mode (the
 * requested size is the 4x target), 1 otherwise.
 */
export const getPidScale = (mode: PidMode): number => (mode === 'native' ? PID_SCALE : 1);

/**
 * The base whose PiD decoder checkpoints are valid for a given main-model base.
 *
 * Decoders are trained per backbone, so only a base-matching decoder may be used.
 * Z-Image is the exception: it shares FLUX.1's 16-channel VAE and ships no decoder of
 * its own, so it reuses the FLUX decoder. Returns null for bases with no PiD support.
 */
export const getPidDecoderBaseForMainBase = (base: string | null | undefined): KnownModelBase | null => {
  switch (base) {
    case 'z-image':
      return 'flux';
    case 'flux':
    case 'flux2':
    case 'sd-3':
    case 'sdxl':
    case 'qwen-image':
      return base;
    default:
      return null;
  }
};

/** Whether a main-model base can decode through PiD. */
export const getIsPidSupportedBase = (base: string | null | undefined): boolean =>
  getPidDecoderBaseForMainBase(base) !== null;

/** Whether PiD is actually engaged: on, and supported by this base. */
export const getIsPidActive = (mode: PidMode, base: string | null | undefined): boolean =>
  mode !== 'off' && getIsPidSupportedBase(base);

const roundDownToMultiple = (value: number, multiple: number): number => Math.floor(value / multiple) * multiple;

/**
 * The resolution to actually denoise at.
 *
 * In native mode the requested size is the 4x target, so generation runs at size / 4,
 * floored onto the model's own grid. Callers snap the requested size to grid * 4 (see
 * `getGenerationDimensions`), so the division lands exactly on the grid; the floor and
 * the `max` only guard hand-entered values.
 */
export const getPidGenerationSize = (
  requested: { width: number; height: number },
  mode: PidMode,
  modelGrid: number
): { width: number; height: number } => {
  if (mode !== 'native') {
    return { height: requested.height, width: requested.width };
  }

  return {
    height: Math.max(roundDownToMultiple(requested.height / PID_SCALE, modelGrid), modelGrid),
    width: Math.max(roundDownToMultiple(requested.width / PID_SCALE, modelGrid), modelGrid),
  };
};

/**
 * The dimension grid and optimal side to present for a PiD mode.
 *
 * Native multiplies the grid by 4 so that requested / 4 stays on the model grid, and
 * reports PiD's own 2048 target as optimal rather than the model's 1024.
 */
export const getPidDimensionOverrides = (
  mode: PidMode,
  base: string | null | undefined,
  modelGrid: number,
  modelOptimalSide: number
): { grid: number; optimalSide: number } => {
  if (!getIsPidActive(mode, base) || mode !== 'native') {
    return { grid: modelGrid, optimalSide: modelOptimalSide };
  }

  return { grid: modelGrid * PID_SCALE, optimalSide: PID_NATIVE_OPTIMAL_SIDE };
};

export const clampPidSteps = (steps: number): number =>
  Math.min(MAX_PID_STEPS, Math.max(MIN_PID_STEPS, Math.round(steps)));
