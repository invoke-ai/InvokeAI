import type { KnownGenerationModelBase as KnownModelBase } from '@features/generation/core/contracts';

import type { PidMode } from './types';

/** Pure PiD geometry and compatibility; pidGraph.ts owns graph wiring. */

/** PiD's fixed super-resolution factor. Every released checkpoint is 4x. */
export const PID_SCALE = 4;

/** Default decode steps. The released checkpoints are 4-step distillations. */
export const DEFAULT_PID_STEPS = 4;
export const MIN_PID_STEPS = 1;
export const MAX_PID_STEPS = 16;

/** The decoder trains from 512 to 2048, making 2048 the native target optimum. */
const PID_NATIVE_OPTIMAL_SIDE = 512 * PID_SCALE;

export const PID_MODES: readonly PidMode[] = ['off', 'fit', 'native'];

export const isPidMode = (value: unknown): value is PidMode =>
  typeof value === 'string' && (PID_MODES as readonly string[]).includes(value);

/** Native PiD scales the target by 4; other modes use 1. */
export const getPidScale = (mode: PidMode): number => (mode === 'native' ? PID_SCALE : 1);

/** Z-Image reuses FLUX's decoder; unsupported bases return null. */
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

/** Divide native target dimensions by four and floor defensively to the model grid. */
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

/** Native PiD uses grid × 4 and target optimum 2048. */
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
